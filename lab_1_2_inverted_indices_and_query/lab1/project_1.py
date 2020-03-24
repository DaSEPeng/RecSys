"""

推荐系统实验1：构建倒排索引

注：
- 这次实验过程与助教提供的参考过程略有差异，过程中多个步骤假设了词数量、文档数量非常多，全部读入内存比较困难等问题
   （emm..不过处理操作不一定非常好）
- Top100 最后的词频数为7，但是词频数为7的有多个，所以在去除TOP100的时候，可能去除的是不同的词，因此最终结果可能不同

@Author：李鹏 (ruhao9805@163.com)
@Date: 2020/3/22

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

def build_inverted_indices(docs):
    """
    为了解耦合并没有在读取文件的时候直接对文件进行处理，这里再次遍历文件构建倒排索引表，过程中有去除高频词操作
    :param docs: 文档列表，格式为[(doc_id_1,['word1','word2',...]),(doc_id_2,['word1','word2',...])]，
                 注意这里假设 doc_id 本身是单调递增的（格式为字符串，如'5','6' ...）
    :return: 倒排索引列表
    """
    print ("Building Inverted Indices ...")
    # 利用词典统计词频，python的字典采用hash表实现，在词的数目比较多的时候效率较高
    # Ref： https://www.cnblogs.com/nxf-rabbit75/p/10566042.html
    term_fre = {}

    # dict中嵌套list，键为一个term，值为去重后的对应的docID list
    # 因为文件读取的docID是单调递增的，所以list中的docID应该已经排好序了
    term_docID = {}

    ## 构建词频表和Term-docID表
    for doc in docs:
        doc_id = doc[0]
        doc_word_list = doc[1]
        for word in doc_word_list:
            # 统计词频
            if word in term_fre:
                term_fre[word] = term_fre[word] + 1
            else:
                term_fre[word] = 1

            # 将doc_ID加入term_docID list
            if word in term_docID:
                if doc_id != term_docID[word][-1]:                   # 去重操作，只需要和末尾id相比较
                    term_docID[word].append(doc_id)
            else:
                term_docID[word] = [doc_id]
    # 现在两个字典的键应该是一样的
    assert term_fre.keys() == term_docID.keys()

    ## 构建前100个高频词的index，后面对这些词进行删除
    # 因为不用和doc_ID列表一块排序，当词对应的doc_ID比较多的话，下面只对词频进行排序的操作速度应该更快，占空间更小
    term_list = [k for k in term_fre.keys()]
    term_fre_list = [v for v in term_fre.values()]
    term_fre_argsort_np = np.argsort(-np.array(term_fre_list),axis=0)
    top_100_index = term_fre_argsort_np[:100]

    ## 对term按照ASCII码进行排序
    term_argsort_np = np.argsort(np.array(term_list),axis=0)

    ## 得到最后的结果
    # 下面的过程很有可能产生跳读，缓存命中率不一定高，但是直接利用sort进行排序也会有缓存命中率的问题
    result_list = []
    for term_index in term_argsort_np:
        if term_index not in top_100_index:
            tmp_term = term_list[term_index]
            tmp_term_fre = term_fre_list[term_index]
            tmp_term_doc_ID_list = term_docID[tmp_term]
            tmp_string = tmp_term + '\t' + str(tmp_term_fre) + '\t' + ' '.join(tmp_term_doc_ID_list) + '\n'
            result_list.append(tmp_string)
    print ("     Inverted Indices Builded!")
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
    # 得到倒排索引表
    result_list = build_inverted_indices(input_file)
    # 将结果写入输出文件
    write_file(output_path,result_list)
    end_time = time.time()
    print ("Time Used: ", end_time - start_time)