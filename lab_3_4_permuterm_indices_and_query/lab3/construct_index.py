"""

推荐系统实验三：构建支持通配查询处理的检索系统---轮排索引的构建
@Author: 李鹏
@Date: 03/31/2020

备注：这里并没有真正实现B Tree版本的索引
"""
import time
import numpy as np

def read_file(file_path,term_lower=True):
    """
    读取文件并进行了预处理，包含大小写转换、字符串划分、文档数统计等操作
    防止文件过大采用 readline()的方式读取，为了防止重新处理一遍文件在读取文件的时候顺便对文件进行了处理
    :param file_path: 输入文件的路径，文件每行为 "doc_id doc_content"
    :return: 文档列表，格式为[(doc_id_1,['word1','word2',...]),(doc_id_2,['word1','word2',...])]
    """
    print ("Reading Data From: ", file_path)
    result = []
    line_num = 0
    with open(file_path,'r') as f:
        while True:
            line = f.readline()
            if line:
                doc_id,doc_content = line.split('\t')
                if term_lower == True:
                    doc_content_list = doc_content.rstrip('\n').lower().split(' ')
                elif term_lower == False:
                    doc_content_list = doc_content.rstrip('\n').split(' ')
                result.append((doc_id,doc_content_list))
                line_num += 1
            else:
                print ("    ",line_num, " Documents Loaded!")
                break
    return result

def rotate_base(str,n):
    """
    该函数可将第1到n个字符整体移动到n+1到len(str) 个字符的右侧。传入不同的n值就可以生成不同的轮排
    :param str: "12345"
    :param n: 2
    :return: "345$12"
    """
    assert(n<len(str))
    str_left = str[0:n]
    str_right = str[n:]
    return str_right + "$" + str_left

def rotate(str):
    """
    得到一个字符串所有的轮排变换List（顺序参考实验要求）
    :param str:
    :return:
    """
    result_list = []
    result_list.append(str + "$")
    result_list.append("$" + str)     # 这两部变换都是固定的
    for i in range(len(str)-1,0,-1):  # 其实是n-1到1 注意这里的顺序
        result_list.append(rotate_base(str,i))
    return result_list

def build_permuterm_indices(docs):
    """
    为了解耦合并没有在读取文件的时候直接对文件进行处理，这里再次遍历文件构建倒排索引表，过程中有去除高频词操作
    :param docs: 文档列表，格式为[(doc_id_1,['word1','word2',...]),(doc_id_2,['word1','word2',...])]，
                 注意这里假设 doc_id 本身是单调递增的（格式为字符串，如'5','6' ...）
    :return: 轮排索引列表
    """
    print ("Building Inverted Indices ...")
    # 利用词典统计词频，python的字典采用hash表实现，在词的数目比较多的时候效率较高
    # Ref： https://www.cnblogs.com/nxf-rabbit75/p/10566042.html
    term_fre = {}

    ## 构建词频表
    for doc in docs:
        # doc_id = doc[0]
        doc_word_list = doc[1]
        for word in doc_word_list:
            # 统计词频
            if word in term_fre:
                term_fre[word] = term_fre[word] + 1
            else:
                term_fre[word] = 1
    ## 构建前100个高频词的index，后面对这些词进行删除(复用了之前的代码)
    term_list = [k for k in term_fre.keys()]
    term_fre_list = [v for v in term_fre.values()]
    term_fre_argsort_np = np.argsort(-np.array(term_fre_list),axis=0)
    top_100_index = term_fre_argsort_np[:100]

    ## 对term按照ASCII码进行排序
    term_argsort_np = np.argsort(np.array(term_list),axis=0)

    ## 得到最后的结果
    result_list = []
    for term_index in term_argsort_np:
        if term_index not in top_100_index:
            tmp_term = term_list[term_index]
            rotate_results =  rotate(tmp_term)
            for r_result in rotate_results:
                tmp_string = r_result + ' ' + tmp_term + '\n'  # 最后一行的换行符暂时没有单独去掉
                result_list.append(tmp_string)
    print ("     Permuterm Indices Builded!")
    return result_list

def write_file(output_path, result_list):
    """
    为了解耦合，将输出到文件的操作单独出来
    :param output_path: 输出文件路径
    :param result_list: 结果列表
    :return: 0
    """
    term_num = 0
    print ("Writing Data to The Path: ", output_path)
    with open(output_path,'w') as f:
        for result in result_list:
            term_num += 1
            f.write(result)
    print ("    ", term_num , "Lines Writed!")
    return 0

if __name__ == '__main__':
    start_time = time.time()
    # 输入输出文件路径
    input_path = './1.txt'
    output_path = './2_generated.txt'
    # 加载文档集
    input_file = read_file(input_path,term_lower=True)
    # 得到轮排索引表
    result_list = build_permuterm_indices(input_file)
    # 将结果写入输出文件
    write_file(output_path,result_list)
    end_time = time.time()
    print ("Time Used: ", end_time - start_time)







