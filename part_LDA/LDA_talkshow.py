# -*- coding: utf-8 -*-
import os
import sys
import traceback
import re
import logging

# ------------------------------
# 全局编码
# ------------------------------
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['LANG'] = 'en_US.UTF-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd
import jieba
import jieba.posseg as pseg
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim_models

# ------------------------------
# 配置开关（白名单模式 + 多样性优化）
# ------------------------------
USE_WHITELIST_ONLY = True      # 启用白名单，但白名单已大幅扩展
NOUNS_ONLY = True              # 仅保留名词
USE_BIGRAM = False

# ------------------------------
# 文件及模型参数（针对600~1000句脱口秀优化）
# ------------------------------
TALK_FILES = [
    'talk_show_sentences.csv',
    'talk_show_sentences(2).csv',
    'talk_show_sentences(1).csv'
    'talk_show_sentences（4）.csv'
]
TEXT_COL = 'Text'
NUM_TOPICS = 6               # 主题数增加至8，细分议题
PASSES = 60
ITERATIONS = 500
NO_BELOW =2                   # 词至少出现6次
NO_ABOVE = 0.55                # 词出现在不超过35%的句子中

OUTPUT_TXT = 'talk_lda_feminist_diverse_output.txt'
OUTPUT_HTML = 'talk_lda_feminist_diverse_vis.html'
MODEL_PREFIX = 'talk_feminist_diverse'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)
try:
    jieba.enable_parallel()
except:
    pass

outfile = open(OUTPUT_TXT, 'w', encoding='utf-8')

feminist_core_words = set([
    '女性', '男性', '性别', '女人', '男人', '女孩', '男孩', '女生', '男生',
    '母亲', '妈妈', '父亲', '爸爸', '女儿', '儿子', '妻子', '丈夫', '老婆', '老公',
    '母亲', '父亲', '孩子', '儿童', '婴儿', '孕妇', '产妇',
    '女性主义', '女权', '女权主义', '性别平等', '性别歧视', '平权', '父权制', '父权',
    '厌女', '厌女症', '男性凝视', '凝视', '物化', '刻板印象', '刻板',
    '身体', '身材', '容貌', '外貌', '颜值', '穿衣自由', '性骚扰', '性侵', '强奸',
    '性同意', '生育', '堕胎', '堕胎权', '代孕', '母职', '母职惩罚', '育儿', '家务',
    '身体自主权', '容貌焦虑', '身材焦虑', '哺乳', '经期', '月经',
    '婚姻', '结婚', '离婚', '婚恋', '嫁娶', '家庭', '家务劳动', '丧偶式育儿',
    '全职妈妈', '家庭主妇', '职场妈妈', '单亲妈妈',
    '职场', '工作', '职业', '就业', '歧视', '玻璃天花板', '经济独立', '独立女性',
    '职场歧视', '同工同酬', '产假', '哺乳假',
    '暴力', '家暴', '家庭暴力', '性暴力', '压迫', '反抗', '觉醒', '挣扎', '自由',
    '权利', '权益', '平等', '独立', '自主', '选择', '声音', '发言权',
    '同性恋', '女同性恋', '拉拉', '同志', 'LGBT', '跨性别', '酷儿', '出柜',
    '芭比', '肯', '父权', '母权', '女巫', '猎巫', 'metoo', '反性骚扰',
    '独立', '自由', '选择', '自主', '觉醒', '反抗', '压迫', '权力', '权利',
'平等', '尊重', '偏见', '刻板', '标签', '束缚', '牺牲', '付出', '照顾',
'职场晋升', '职业发展', '工资', '晋升', '领导', '决策', '声音', '表达',
'照顾家庭', '带娃', '家务分工', '丧偶', '出轨', '冷暴力', '精神控制',
'离婚冷静期', '彩礼', '嫁妆', '重男轻女', '生儿子', '传宗接代',
'卫生巾', '痛经', '更年期', '绝经', '月经羞耻',
'女孩教育', '女童', '失学', '早婚', '生育机器',
'化妆自由', '服美役', '身材管理', '减肥', '整容',
'同工同酬', '产假', '哺乳假', '育儿假',
'性教育', '避孕', '人流', '药流', '紧急避孕',
'女博士', '女强人', '剩女', '大龄未婚',
    '女性', '男性', '性别', '女人', '男人', '女孩', '男孩', '女生', '男生',
    '母亲', '妈妈', '父亲', '爸爸', '女儿', '儿子', '妻子', '丈夫', '老婆', '老公',
    '孩子', '儿童', '婴儿', '孕妇', '产妇',
    '女性主义', '女权', '女权主义', '性别平等', '性别歧视', '平权', '父权制', '父权',
    '厌女', '厌女症', '男性凝视', '凝视', '物化', '刻板印象', '刻板',
    '身体', '身材', '容貌', '外貌', '颜值', '穿衣自由', '性骚扰', '性侵', '强奸',
    '性同意', '生育', '堕胎', '堕胎权', '代孕', '母职', '母职惩罚', '育儿', '家务',
    '身体自主权', '容貌焦虑', '身材焦虑', '哺乳', '经期', '月经',
    '婚姻', '结婚', '离婚', '婚恋', '嫁娶', '家庭', '家务劳动', '丧偶式育儿',
    '全职妈妈', '家庭主妇', '职场妈妈', '单亲妈妈',
    '职场', '工作', '职业', '就业', '歧视', '玻璃天花板', '经济独立', '独立女性',
    '职场歧视', '同工同酬', '产假', '哺乳假',
    '暴力', '家暴', '家庭暴力', '性暴力', '压迫', '反抗', '觉醒', '挣扎', '自由',
    '权利', '权益', '平等', '独立', '自主', '选择', '声音', '发言权',
    '同性恋', '女同性恋', '拉拉', '同志', 'LGBT', '跨性别', '酷儿', '出柜',
    '芭比', '肯', '母权', '女巫', '猎巫', 'metoo', '反性骚扰',
    '晋升', '升职', '加薪', '工资', '薪酬', '收入', '失业', '待业',
    '领导', '管理', '高管', '决策', '开会', '出差', '加班', '996',
    '面试', '简历', '招聘', '性别比', '男女比例', '工科', '理科',
    '创业', '老板', '上司', '同事', '团队', '合作', '职场霸凌',
    '彩礼', '嫁妆', '婚房', '户口', '生育率', '二胎', '三胎',
    '催婚', '相亲', '逼婚', '单身', '剩女', '大龄', '不婚',
    '丁克', '分居', '出轨', '婚外情', '小三', '离婚冷静期',
    '抚养权', '赡养', '养老', '婆媳', '妯娌', '大男子主义',
    '贤惠', '温顺', '听话', '懂事', '牺牲', '奉献', '付出', '包容',
    '减肥', '瘦身', '苗条', '体重', '饮食', '节食', '健身',
    '化妆', '素颜', '口红', '穿搭', '医美', '整容', '隆胸',
    '美白', '防晒', '脱毛', '姨妈', '痛经', '更年期', '绝经',
    '妇科', '乳腺', '子宫', '卵巢', 'HPV', '宫颈', '避孕药',
    '卫生巾', '月经杯', '月嫂', '催乳', '产后抑郁',
    '重男轻女', '传宗接代', '续香火', '养儿防老', '偏心',
    '性别刻板', '标签', '污名', '荡妇羞辱', '受害者有罪论',
    '男权', '女尊', '平权', '激进', '保守', '刻板印象',
    '偏见', '歧视', '优待', '逆歧视', '反歧视', '权益', '维权',
    '上学', '辍学', '升学', '高考', '专业', '文科', '理科',
    '学历', '硕士', '博士', '女博士', '学霸', '学渣',
    '早恋', '青春期', '叛逆', '家庭教育', '子女教育',
    '强奸犯', '猥亵', '偷拍', '性侵害', '暴力', '殴打', '虐待',
    '冷暴力', '精神控制', 'PUA', '恐吓', '跟踪', '尾随',
    '报警', '诉讼', '法律', '庇护所', '保护令',
    '报道', '新闻', '热搜', '舆论', '公知', '大V', '媒体',
    '言论', '禁言', '删帖', '封号', '举报', '意识形态',
    '女权运动', '游行', '集会', '请愿', '联名', '罢工',
    '志愿者', '公益', 'NGO', '基金会', '奖学金', '互助',
    '娘炮', '伪娘', '女汉子', '女装', '男装', '中性',
    '柔弱', '坚强', '勇敢', '脆弱', '刚强', '温柔',
    '霸道', '任性', '撒娇', '卖萌', '绿茶', '白莲花',
    '小说', '影视', '剧集', '角色', '女主角', '男主角',
    '剧情', '编剧', '导演', '作家', '书评', '影评',
])

# ------------------------------
# 停用词表（通用 + 脱口秀特有，已大幅精简以突出重点）
# ------------------------------
extra_stop_words = [
    # 以下仅列出最关键的口语词和泛词，您可以根据需要补充
    '观众', '大家', '各位', '朋友', '现场', '台下', '台上', '哈哈', '哈哈哈',
    '呵呵', '嘿嘿', '掌声', '笑', '逗', '搞笑', '段子', '梗', '包袱', '脱口秀',
    '单口', '漫才', '吐槽大会', '脱口秀大会', '奇葩说', '演员', '表演', '舞台',
    '节目', '嘉宾', '主持', '开场', '收尾', '互动', '观众反应', '效果',
    '然后', '其实', '真的', '有点', '算是', '感觉', '就是', '每次', '今天',
    '昨天', '明天', '开始', '结束', '已经', '还是', '不过', '所以', '但是',
    '而且', '做', '说', '看', '听', '想', '觉得', '认为', '以为', '感到',
    '时间', '地方', '事情', '东西', '方面', '情况', '结果', '第一', '第二',
    '第三', '首先', '其次', '最后', '总之', '总的来说', '还有', '等等',
    '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
    '上', '也', '很', '到', '要', '去', '你', '会', '着', '没有', '好',
    '自己', '这', '那', '但', '可以', '这个', '那个', '什么', '怎么',
    '因为', '所以', '如果', '还是', '可能', '知道', '里面', '出来',
    '一样', '时候', '然后', '不能', '不是', '就是', '比较', '这么', '那么',
    '一点', '有点', '一些', '很多', '不过', '之后', '以前', '只能', '只是',
    '的话', '啊', '吧', '哦', '嗯', '呢', '吗', '呀', '啦', '我们', '你们',
    '他们', '她们', '但是', '最后', '真是', '不错', '不要', '问题', '后面',
    '感动', '需要', '生活', '前面', '还有', '时候', '时间', '地方', '事情',
]
stop_words = set(extra_stop_words)

# ------------------------------
# 同义词合并（核心关系词）
# ------------------------------
synonym_dict = {
    '女性': ['女人', '女生', '女孩', '女子', '女性们', '姑娘', '女士'],
    '男性': ['男人', '男生', '男孩', '男子', '男性们', '小伙子', '男士'],
    '母亲': ['妈妈', '妈妈们', '母亲们', '宝妈', '妈咪', '老妈'],
    '父亲': ['爸爸', '父亲们', '宝爸', '爹地', '老爸'],
    '女儿': ['闺女', '女儿们', '丫头'],
    '儿子': ['儿子们', '小子'],
    '妻子': ['老婆', '媳妇', '太太', '夫人'],
    '丈夫': ['老公', '先生', '爱人'],
    '孩子': ['小孩', '小朋友', '孩子们', '宝宝', '婴儿', '娃', '娃儿'],
    '性别': ['男女', '性别平等', '性别差异'],
    '女性主义': ['女权', '女权主义', '妇女解放'],
    '身体': ['身材', '容貌', '长相', '外貌', '颜值'],
    '婚姻': ['结婚', '离婚', '婚恋', '嫁娶', '婚姻关系'],
    '家庭': ['家', '家人', '家庭生活', '家务'],
}

def merge_synonyms(word):
    for core, syns in synonym_dict.items():
        if word in syns:
            return core
    return word

# 添加脱口秀常见人名和术语到词典
custom_words = [
    '女性主义', '厌女', '男性凝视', '穿衣自由', '独立女性',
    '家庭主妇', '全职妈妈', '母职惩罚', '性骚扰', '性同意',
    '性别平等', '性别歧视', '父权制', '生育自由', '堕胎权',
    '职场歧视', '玻璃天花板', '经济独立', '精神独立', '身体自主权',
    '女权主义', '平权', '母职', '代孕', '性侵', 'metoo', '反性骚扰',
    '芭比', '肯', '厌女症', '丧偶式育儿', '家务劳动', '同工同酬',
    '杨笠', '呼兰', '庞博', '周奇墨', '李诞', '王建国', '赵晓卉', 'ROCK',
]
for w in custom_words:
    jieba.add_word(w)

# ------------------------------
# 预处理与分词
# ------------------------------
def preprocess_text(text):
    text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'[a-zA-Z0-9]+', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    return text

def tokenize_talk(text):
    clean = preprocess_text(text)
    words_pos = pseg.cut(clean)
    tokens = []
    for word, flag in words_pos:
        if len(word) <= 1:
            continue
        if word in stop_words:
            continue
        if USE_WHITELIST_ONLY:
            if word in feminist_core_words or merge_synonyms(word) in feminist_core_words:
                word = merge_synonyms(word)
                tokens.append(word)
        else:
            if NOUNS_ONLY:
                if flag.startswith('n'):
                    word = merge_synonyms(word)
                    tokens.append(word)
            else:
                if flag.startswith(('n', 'v', 'a', 'i', 'l', 'an')):
                    word = merge_synonyms(word)
                    tokens.append(word)
    return tokens

# ------------------------------
# 加载数据
# ------------------------------
def load_and_merge_talk_files(file_list, text_col):
    dfs = []
    for path in file_list:
        try:
            df = pd.read_csv(path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding='utf-8-sig')
        except FileNotFoundError:
            print(f"警告：文件 {path} 不存在，跳过")
            continue
        dfs.append(df)
        print(f"  加载 {path}: {len(df)} 行")
    if not dfs:
        raise FileNotFoundError("未找到任何脱口秀文件")
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=[text_col], keep='first')
    return combined

# ------------------------------
# LDA 训练与输出
# ------------------------------
def run_lda_talk(df, text_col, num_topics, passes):
    print("开始训练脱口秀 LDA（白名单模式 + 多样性优化）...", flush=True)
    outfile.write(f"\n{'='*60}\n脱口秀 LDA 主题分析（白名单模式，多样性优化）\n{'='*60}\n")
    outfile.write(f"白名单词汇数: {len(feminist_core_words)}\n")
    outfile.write(f"主题数: {num_topics}, NO_BELOW={NO_BELOW}, NO_ABOVE={NO_ABOVE}\n\n")

    texts = df[text_col].apply(tokenize_talk).tolist()
    texts = [t for t in texts if len(t) >= 2]
    outfile.write(f"有效句子数（含至少2个白名单词）: {len(texts)}\n")
    print(f"有效句子数: {len(texts)}")

    if len(texts) < 50:
        outfile.write("警告：有效句子数过少，请检查白名单或降低过滤条件。\n")

    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE)
    dictionary.compactify()
    corpus = [dictionary.doc2bow(text) for text in texts]

    # 关键：使用 eta=0.01 增加主题区分度，alpha='auto' 保留自然多样性
    lda_model = models.LdaModel(
        corpus, id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        iterations=ITERATIONS,
        random_state=42,
        eval_every=None,
        alpha='auto',
        eta=0.01
    )

    outfile.write("\n各主题 top-15 关键词:\n")
    for topic_id, topic_words in lda_model.print_topics(num_words=15):
        outfile.write(f"主题 {topic_id}: {topic_words}\n")
        print(f"主题 {topic_id}: {topic_words[:100]}...")

    # 计算 coherence 评估主题质量
    try:
        from gensim.models.coherencemodel import CoherenceModel
        coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence = coherence_model.get_coherence()
        outfile.write(f"\n整体主题 coherence (c_v): {coherence:.4f}\n")
        print(f"Coherence: {coherence:.4f}")
    except Exception as e:
        outfile.write(f"Coherence 计算失败: {e}\n")

    outfile.write("\n每个主题的高置信度句子样本（概率>0.5）:\n")
    doc_topics = lda_model.get_document_topics(corpus, minimum_probability=0.3)
    for topic_id in range(num_topics):
        topic_docs = []
        for i, dist in enumerate(doc_topics):
            for t, prob in dist:
                if t == topic_id and prob > 0.5:
                    topic_docs.append((i, prob, df[text_col].iloc[i]))
        topic_docs.sort(key=lambda x: x[1], reverse=True)
        outfile.write(f"\n--- 主题 {topic_id} ---\n")
        for i, prob, doc_text in topic_docs[:3]:
            outfile.write(f"  (概率={prob:.2f}) {doc_text[:200]}...\n")

    try:
        vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary,
                                                  n_jobs=1, lambda_step=0.01, sort_topics=False)
        with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
            pyLDAvis.save_html(vis_data, f)
        outfile.write(f"\n可视化保存至 {OUTPUT_HTML}\n")
        print(f"✅ 可视化成功: {OUTPUT_HTML}")
    except Exception as e:
        outfile.write(f"可视化失败: {e}\n")
        traceback.print_exc(file=outfile)

    lda_model.save(f'{MODEL_PREFIX}.model')
    dictionary.save(f'{MODEL_PREFIX}_dict.model')
    return lda_model

# ------------------------------
# 主程序
# ------------------------------
def main():
    try:
        print("="*60)
        print("脱口秀文本 - 女性主义聚焦 LDA（白名单 + 多样性优化）")
        print(f"白名单模式: {USE_WHITELIST_ONLY}, 主题数: {NUM_TOPICS}")
        df = load_and_merge_talk_files(TALK_FILES, TEXT_COL)
        outfile.write(f"合并后总句子数（去重）: {len(df)}\n")
        print(f"合并后总句子数: {len(df)}")
        lda = run_lda_talk(df, TEXT_COL, NUM_TOPICS, PASSES)
        print("\n🎉 分析完成！")
        print(f"📄 结果: {OUTPUT_TXT}")
        print(f"📊 可视化: {OUTPUT_HTML}")
    except Exception as e:
        outfile.write(f"错误: {e}\n")
        traceback.print_exc(file=outfile)
        print(f"❌ 出错: {e}")
    finally:
        outfile.close()
        try:
            jieba.disable_parallel()
        except:
            pass

if __name__ == '__main__':
    main()