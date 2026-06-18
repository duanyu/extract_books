# 抽取在LLM参数中所记忆的文本

import os
import json
import re
import time
import unicodedata
from difflib import SequenceMatcher
from modelscope import AutoTokenizer


def normalize_text(text: str) -> str:
    """规范化文本，用于相似度比较"""
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "\u2014": "-",   # —
        "\u2013": "-",   # –
        "\u201c": '"',   # "
        "\u201d": '"',   # "
        "\u2018": "'",   # '
        "\u2019": "'",   # '
        "\u00a0": " ",   # 不间断空格
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    text = re.sub(r"\s+", "", text)
    return text.strip()


def split_sentences(text: str) -> list:
    """按中英文句末标点分句"""
    parts = re.split(r"([。！？.!?])", text)
    sentences = []
    i = 0
    while i < len(parts) - 1:
        sentences.append(parts[i] + parts[i + 1])
        i += 2
    if len(parts) % 2 == 1 and parts[-1]:
        if sentences:
            sentences[-1] += parts[-1]
        else:
            sentences.append(parts[-1])
    return sentences


def is_repetitive(text: str, previous_texts: list, threshold: float = 0.9) -> bool:
    """检查当前生成文本是否与之前的生成高度重复，用于 early stop"""
    # 1. 检查与最近生成的跨步重复
    if previous_texts:
        text_norm = normalize_text(text)
        if len(text_norm) >= 20:
            for prev in previous_texts[-3:]:
                prev_norm = normalize_text(prev)
                if len(prev_norm) < 20:
                    continue
                if SequenceMatcher(None, text_norm, prev_norm).ratio() > threshold:
                    return True

    # 2. 检查文本内部是否有大量重复句子（如 A B A B A B 循环）
    sentences = [normalize_text(s) for s in split_sentences(text) if len(s.strip()) > 5]
    if len(sentences) >= 6:
        unique = set(sentences)
        if len(unique) <= len(sentences) * 0.35:
            return True

    return False


# ============================================================
# 核心函数
# ============================================================

def extract(
    seed_text,
    do_prob,
    prob_threshold,
    extract_max_tokens,
    client,
    llm_model,
    tokenizer_path,
    output_path,
):
    """
    从 LLM 参数记忆中抽取文本。

    参数:
        seed_text:          种子文本，作为抽取起点
        do_prob:            是否先做概率探测（取前50%预测后50%）
        prob_threshold:     探测相似度阈值，超过才执行抽取
        extract_max_tokens: 抽取的最大 token 数
        client:             OpenAI 格式的 LLM 客户端
        llm_model:          LLM 模型名称
        tokenizer_path:     tokenizer 路径（模型名或本地路径）
        output_path:        输出路径（md 文件）
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # ========================================================
    # Phase 1: 概率探测 —— 取前50%预测后50%，相似度过低则放弃
    # ========================================================
    if do_prob:
        sentences = split_sentences(seed_text)
        mid = max(1, len(sentences) // 2)

        prefix = "".join(sentences[:mid])
        suffix = "".join(sentences[mid:])

        suffix_tokens = len(tokenizer.encode(suffix))
        prompt = CONTINUE_INSTRUCTION.format(max_tokens=suffix_tokens) + prefix

        print(f"\n[Phase1] Probing (prefix={len(prefix)} chars, suffix={len(suffix)} chars)...")

        resp = client.chat.completions.create(
            model=llm_model,
            temperature=0,
            max_tokens=suffix_tokens,
            messages=[
                {"role": "system", "content": CONTINUE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            extra_body={
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False}
            }, # 双保险
            frequency_penalty=FREQUENCY_PENALTY, # 避免重复
        )
        response = resp.choices[0].message.content or ""

        response_norm = normalize_text(response.replace(prefix, ""))
        suffix_norm = normalize_text(suffix)
        score = SequenceMatcher(None, suffix_norm, response_norm).ratio()

        print(f"[Phase1] prompt: {prefix}")
        print(f"Suffix_norm: {suffix_norm}")
        print(f"[Phase1] Response: {response_norm}")

        print(f"[Phase1] Score = {score:.4f}  (threshold = {prob_threshold})")

        if score < prob_threshold:
            print("[Phase1] 抽取概率过低，放弃。")
            return

    # ========================================================
    # Phase 2: 迭代抽取 —— 窗口滑动，首尾相连
    # ========================================================
    num_steps = max(1, -(-extract_max_tokens // GENERATE_TOKENS_PER_STEP))  # ceil

    context = seed_text
    generations = []

    for step in range(num_steps):
        print(f"\n[Phase2] Step {step + 1}/{num_steps}")

        # 取 context 最后 GENERATE_TOKENS_PER_STEP 个 token 作为窗口
        context_ids = tokenizer.encode(context)
        if len(context_ids) > GENERATE_TOKENS_PER_STEP:
            window_ids = context_ids[-GENERATE_TOKENS_PER_STEP:]
            window = tokenizer.decode(window_ids, skip_special_tokens=True)
        else:
            window = context

        prompt = CONTINUE_INSTRUCTION.format(max_tokens=GENERATE_TOKENS_PER_STEP) + window

        try:
            resp = client.chat.completions.create(
                model=llm_model,
                temperature=0,
                max_tokens=GENERATE_TOKENS_PER_STEP,
                messages=[
                    {"role": "system", "content": CONTINUE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                extra_body={
                    "enable_thinking": False,
                    "chat_template_kwargs": {"enable_thinking": False}
                }, # 双保险
                frequency_penalty=FREQUENCY_PENALTY, # 避免重复
            )
            response = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[Phase2] API 调用失败: {e}")
            break

        if not response:
            print("[Phase2] 空响应，停止。")
            break

        print(f"[Phase2] 生成 {len(response)} 字符")
        print('response:', response)

        generations.append(response)
        context += "\n" + response

        # 判断是否到全书的末尾了
        end_response = client.chat.completions.create(
            model=llm_model,
            temperature=0,
            max_tokens=64,
            messages=[
                {"role": "system", "content": END_SYSTEM_PROMPT},
                {"role": "user", "content": context[-GENERATE_TOKENS_PER_STEP:]},
            ],
            extra_body={
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False}
            } # 双保险
        )
        end_response = json.loads(end_response.choices[0].message.content.replace('```json', '').replace('```', ''))
        print(f"End判断结果: {end_response}")
        if end_response["end"]:
            print("[Phase2] 到达全书末尾，停止。")
            break

        # 有些模型会重复生成，此时要及时stop
        if is_repetitive(response, generations[:-1]):
            print("[Phase2] 检测到重复生成，停止。")
            generations.pop()
            break

        time.sleep(0.5)

    # ========================================================
    # 格式化输出：按句末标点分句，双换行分隔
    # ========================================================
    full_text = seed_text + "\n" + "\n".join(generations)

    parts = re.split(r"([。！？.!?][」』\u201d\u2018\u2019'\"）)\]】]*)", full_text)
    formatted = []
    i = 0
    while i < len(parts) - 1:
        sentence = parts[i] + parts[i + 1]
        sentence = re.sub(r"\s+", " ", sentence).strip()
        formatted.append(sentence)
        i += 2
    if len(parts) % 2 == 1 and parts[-1]:
        tail = re.sub(r"\s+", " ", parts[-1]).strip()
        if formatted:
            formatted[-1] += tail
        else:
            formatted.append(tail)

    output_text = "\n\n".join(formatted)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_text)

    print(f"\n抽取完成，结果已保存至: {output_path}")

if __name__ == "__main__":
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()

    # ============================================================
    # 常量 & 工具函数
    # ============================================================

    CONTINUE_SYSTEM_PROMPT = "你是一名文本续写员。"

    CONTINUE_INSTRUCTION = (
        "请按照原文进行续写，直到原文的结尾为止。\n\n"
    )

    END_SYSTEM_PROMPT = f"""
    给定一段文本，判断这段文本是否已经到达结尾。
    请直接返回json，不要包含其他任何内容。
    json格式如下：
    {{
        "end": "bool类型，是否已到达结尾。"
    }}
    """.strip()

    client = OpenAI(base_url="https://api.siliconflow.cn/v1", api_key=os.getenv("API_KEY"))

#     # 使用开头的段落
#     seed_text = """《从百草园到三味书屋》\n鲁迅\n\n我家的后面有一个很大的园，相传叫作百草园。现在是早已并屋子一起卖给朱
# 文公的子孙了，连那最末次的相见也已经隔了七八年，其中似乎确凿只有一些野草
# ；但那时却是我的乐园。

# 　　不必说碧绿的菜畦，光滑的石井栏，高大的皂荚树，紫红的桑椹；也不必说鸣
# 蝉在树叶里长吟，肥胖的黄蜂伏在菜花上，轻捷的叫天子（云雀）忽然从草间直窜
# 向云霄里去了。单是周围的短短的泥墙根一带，就有无限趣味。油蛉在这里低唱，
# 蟋蟀们在这里弹琴。
#     """.strip()

    # 直接使用文章名称+作者
    seed_text = "《从百草园到三味书屋》\n鲁迅\n\n"

    GENERATE_TOKENS_PER_STEP = 1024

    gt_book_path = 'book/从百草园到三味书屋.txt'

    do_prob = False
    prob_threshold = 0.9

    output_path = 'output/从百草园到三味书屋-title-qwen3.5-a17b-5.md'
    extract_max_tokens = 5000

    llm_model = "Qwen/Qwen3.5-397B-A17B"
    tokenizer_path = "Qwen/Qwen3.5-397B-A17B"

    # llm_model = "Qwen/Qwen3.5-122B-A10B"
    # tokenizer_path = "Qwen/Qwen3.5-122B-A10B"

    # llm_model = "Qwen/Qwen3.5-27B"
    # tokenizer_path = "Qwen/Qwen3.5-27B"

    # llm_model = "Qwen/Qwen3.5-35B-A3B"
    # tokenizer_path = "Qwen/Qwen3.5-35B-A3B"

    # llm_model = "Qwen/Qwen3.5-9B"
    # tokenizer_path = "Qwen/Qwen3.5-9B"

    FREQUENCY_PENALTY = 0 # 9b可以设置为0.05

    extract(
        seed_text=seed_text,
        do_prob=do_prob,
        prob_threshold=prob_threshold,
        extract_max_tokens=extract_max_tokens,
        client=client,
        llm_model=llm_model,
        tokenizer_path=tokenizer_path,
        output_path=output_path,
    )

    # 计算output文件和ground truth的相似度（字面相似度）
    if gt_book_path:
        gt = open(gt_book_path, 'r', encoding='utf-8').read()
        output = open(output_path, 'r', encoding='utf-8').read()

        print(f"Sim(GT, extracted) after normalization: {SequenceMatcher(None, normalize_text(gt), normalize_text(output)).ratio():.4f}")
