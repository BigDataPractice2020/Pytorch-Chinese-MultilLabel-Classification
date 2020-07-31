import pandas as pd
import numpy as np
from copy import deepcopy
from m2_dict import ai_map,in_dct,out_dct
import jieba
import os


def del_duplicate(info, info_name):
    """
    # 删除重复的文本 在同一个数据集下
    """
    # ['processid', 'in_node', 'type_robot', 'msg', 'msg_del_dup', 'out_true', 'type', 'type_combine']
    set_only = set()
    new_info = []
    for item in info:
        msg = item[3]
        if msg not in set_only:
            set_only.add(msg)
            new_info.append(item)
    print("\n对 {} 去重, 被删除的重复的语料的数量（以msg为准）: {}. Final number: {}"
          .format(info_name, len(info) - len(new_info), len(new_info)))
    return np.array(new_info)


def build_map_by_what(info, what_col):
    # 按照意图节点切分: what_col = 2; 按照type切分: what_col = 6
    # print(info)
    info = np.array(info)
    nodes = np.unique(np.array(info[:, what_col]))
    map_temp = dict(zip(nodes, [[] for _ in range(len(nodes))]))
    for item in info:
        map_temp[item[what_col]].append(item)
    return map_temp


def read_data(file_dir):
    cols = ["通话状态", "通话记录id","通话记录"]
    all_data = []
    for root, dirs, files in os.walk(file_dir):
        for curr_file in files:
            print("read file: {}".format(os.path.join(root, curr_file)))
            temp = pd.read_csv(os.path.join(root, curr_file), usecols=cols).values
            print("---------------------------------")
            print(temp)
            all_data += list(temp)
    all_data = [x for x in all_data if x[1] == '已接听' and isinstance(x[2], str) and len(x[2]) != 0 ]
    return all_data


def split_ai_me(data,ai_map,wu_dic,out_dic):
    in_node = []
    type_robot = []
    me = []
    out_node = []
    type = []
    AI = []
    S_id = [] #session_id
    for line in data:
        # ["通话记录ID","通话状态", "通话记录详情"]
        texts = line[2].split('\n')
        s_id = str(line[0]).strip('')

        # 初始化index count
        index= 0
        pre_ai, rear_ai = '', '' #in_node 和 out_node
        pre_ai_key,rear_ai_key='*','**'
        # 遍历文本：AI 和 ME
        while index < len(texts):
            temp = texts[index]
            if not texts[index].startswith('ME'): #当前的ai问题是什么 in_node
                if texts[index].startswith('AI'):
                    tt = texts[index]
                    for key_word in ai_map.keys():
                        if key_word in texts[index]:
                            pre_ai = ai_map[key_word]
                            pre_ai_key=key_word
                            break
                        pre_ai_key='*' #有AI说话无关键词
                index += 1
                continue
            while texts[index].startswith('ME'):
                kk = texts[index]
                # 当前标签
                index_ai = index
                while index_ai < len(texts) and (not texts[index_ai].startswith('AI')):
                    index_ai += 1
                if index_ai < len(texts) and texts[index_ai].startswith('AI'):# me 回答后的ai问题是什么 out_node
                    for key_word in ai_map.keys():
                        if key_word in texts[index_ai]:
                            rear_ai = ai_map[key_word]
                            rear_ai_key = key_word
                            break
                        rear_ai_key = '*' #有AI说话但无关键字
                in_node.append(wu_dic[pre_ai_key][0])
                type_robot.append(wu_dic[pre_ai_key][1])
                me.append(texts[index][3:])
                out_node.append(out_dic[rear_ai_key][0])
                type.append(out_dic[rear_ai_key][1])

                AI.append(tt)
                S_id.append(s_id)
                rear_ai_key ='**' #重置
                # index加一：下一条AI或ME
                index += 1

    # 把 p_id in_node type_robot me out_node type 放在一起
    ans, temp = [], []
    for index in range(len(in_node)):
        if type_robot[index] != "":
            temp.append('hayinm2')
            temp.append(in_node[index])
            temp.append(type_robot[index])
            temp.append(me[index])
            temp.append(out_node[index])
            temp.append(type[index])
            temp.append(AI[index])
            temp.append(S_id[index])
            ans.append(deepcopy(temp))
            temp.clear()
    return ans


def del_dul_word(dul_word):
    new_string = []
    pre_ch = None
    for ch in dul_word:
        if ch != pre_ch:
            new_string.append(ch)
            pre_ch = ch
    return new_string


def insert_cols(data):
    jieba.load_userdict('./lexicon_external.txt')
    ans = []
    for item in data:
        msg = item[3].strip()
        # 仅保留中文
        msg = ''.join([ch for ch in msg if ('\u4e00' <= ch <= '\u9fa5')])
        if len(msg) == 0:
            continue
        item[3] = msg
        # 去叠词：msg_del_dul
        msg_del_dul = del_dul_word([word for word in jieba.cut(msg)])
        item.append(''.join(msg_del_dul))
        # item += ['']
        item.append('')
        ans.append(item)
    print('过滤 msg左右空格+符号+英文+数字 仅留中文汉字: {} -> {}'.format(len(data), len(ans)))
    return ans


if __name__ == '__main__':
    input_dir = './input_m2_4-10_1000'
    high_data = read_data(input_dir)


    # 处理 "通话记录详情"
    high_data = split_ai_me(high_data,ai_map,in_dct,out_dct)


    #插入新列：去叠词
    high_data = insert_cols(high_data)


    # 去重复
    high_data_by_node = build_map_by_what(high_data, 2)
    del high_data
    high_data = []
    for key_node, val_node in high_data_by_node.items():
        high_data += list(del_duplicate(val_node, key_node))
        print("当前数量量：{}".format(len(high_data)))
    # del high_data_by_node
    # 保存
    str = 'hayinm2duolun' # 直接更改文件名标识
    sava_path = './output_m2_1000/'+str+'_result.csv'
    original_col = ["processid", "in_node", "type_robot", "msg", "out_true", "type", "AI_Q","session_id", "msg_del_dup", "type_combine"]
    new_col = ["session_id","processid", "AI_Q", "in_node", "type_robot", "msg", "msg_del_dup", "out_true", "type", "type_combine"]
    #所有数据保存
    high_data = pd.DataFrame(high_data, columns=original_col)
    high_data = high_data.reindex(columns=new_col)
    high_data.to_csv(sava_path, encoding='utf-8-sig', index_label='id')

    # # 划分：需要标注 and 不需要标注/入结点知识库+表示结束
    tag ,un_tag = [], []
    for item in high_data.values:
        if item[3] in ['无AI其他','知识库','敏感词','其他','3.1','3.3','4.4','5.1/5.3','5.2/5.4','5.5','5.6'] :
            un_tag.append(item)
        else:
            tag.append(item)

    # 需要标注
    tag = sorted(tag, key=lambda k: (k[4]))
    tag=pd.DataFrame(tag,columns=new_col)
    tag.to_csv('./output_m2_1000/'+str+'_tag.csv',encoding='utf-8-sig', index_label='id')

    # 不需要标注
    un_tag = sorted(un_tag, key=lambda k: (k[4]))
    un_tag = pd.DataFrame(un_tag, columns=new_col)
    un_tag.to_csv('./output_m2_1000/'+str+'_untag.csv', encoding='utf-8-sig', index_label='id')