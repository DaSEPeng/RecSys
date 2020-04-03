"""

推荐系统实验4：轮排索引查询
@Author：李鹏 (ruhao9805@163.com)
@Date: 2020/4/3

备注：考虑到数据处理方便，以及实际生产中的需要，这里的重用了实验1和实验3中的结果数据，即倒排索引表和轮排对照表
"""

import time

def load_inverted_indices(file_path):
    """
    加载倒排索引列表，得到一个元组，元组的第一个元素为term list，元组的第二个元素为docID嵌套list，相同位置的term和docID list相互对应
    :param file_path: 倒排索引文件位置
    :return: 元组
    """
    print ("Loading Indices ...")
    index_num = 0
    term_list = []
    docID_list = []
    with open(file_path,'r') as f:
        while True:
            line = f.readline()
            if line:
                term,term_fre,doc_ids = line.rstrip('\n').split('\t')         # [:-1]是为了去除换行符
                doc_ids = doc_ids.split(' ')
                term_list.append(term)                         # term_fre在只有一个AND或者OR的查询中没有太大作用，不需要
                docID_list.append(doc_ids)
                index_num +=1
            else:
                print ("    Indices Num: ",index_num)
                break
    print ("    Indices Loaded!")
    return (term_list,docID_list)

def load_permuterm_indices(file_path):
    """
    读取轮排索引对照表，返回对应的倒排索引term（已经按照倒排变换结果即第一列排好序）
    :param file_path:
    :return:
    """
    print ("Loading Permuterm Indices ...")
    result_list = []
    permuterm_num = 0
    with open(file_path,'r') as f:
        while True:
            line = f.readline()
            if line:
                line = line.rstrip('\n').split(' ')
                if len(line) > 1:
                    result_list.append([line[0],line[1]]) # 元组，（轮排结果，原始term）
                permuterm_num += 1
            else:
                print ("    {} Indices Loaded!".format(permuterm_num))
                break
    result_list = sorted(result_list, key=lambda x: x[0])   # 按照第一列升序排序
    return result_list

def load_queries(file_path):
    """
    加载queries数据，得到去除换行符之后的query列表
    :param file_path: 文件路径
    :return: query列表
    """
    print("Loading Queries ...")
    query_num = 0
    result = []
    with open(file_path,'r') as f:
        while True:
            line = f.readline()
            if line:
                result.append(line.rstrip('\n'))
                query_num += 1
            else:
                print ("    Queries Num: ", query_num)
                break
    print("    Queries Loaded!")
    return result

def query_indices_base(tmp_term,tmp_indices):
    """
    得到一个term对应的docID列表，若没有查询到，便返回空列表
    :param tmp_term:当前term
    :param tmp_indices: 倒排索引表，包含term_list和docID_list两个部分
    :return:当前term对应的docID列表
    """
    term_list = tmp_indices[0]
    docID_list = tmp_indices[1]
    try:
        index = term_list.index(tmp_term)
        result_docID_list = [eval(i) for i in docID_list[index]]
    except:
        result_docID_list = []
    return result_docID_list

def query_permuterm_indices(query,permuterm_indices):
    """
    查询轮排indices（实验3结果）中轮排项前缀是_query的term项
    :param _query:
    :param permuterm_indices:
    :return:
    """
    result = []
    start = 0   # 标记是否已经找到开始的了
    query_len = len(query)
    for indices in permuterm_indices:
        permuterm = indices[0]
        ind_len = len(permuterm)
        if ind_len >= query_len:
            if query == permuterm[:query_len]:
                start = 1
                result.append(indices[1])  # 返回对应的term
            elif query > permuterm and start == 1:    # 因为已经排好序了，这里能加速
                print (query)
                print (permuterm)
                print ("\n")
                break
    return result

def query_indices(tmp_query,permuterm_indices,tmp_indices):
    """
    处理一个query
    :param tmp_query: 当前的query
    :param permuterm_indices: 轮排对照表
    :param tmp_indices: 倒排索引表
    :return: 查询结果
    """
    tmp_query_splited = tmp_query.split('*')  # 每一行都有*，肯定有两个元素

    # 得到轮排后的查询，这里是前缀，省略了*号
    if tmp_query_splited[1] == '':        # X*
        _query = "$" + tmp_query_splited[0]
    elif tmp_query_splited[0] == '':      # *X
        _query = tmp_query_splited[1] + "$"
    elif tmp_query_splited[0] != '' and tmp_query_splited[1] != '':  # X*Y
        _query = tmp_query_splited[1] + "$" + tmp_query_splited[0]

    term_list = query_permuterm_indices(_query,permuterm_indices)
    result_list = []
    for term in term_list:
        tmp_docID_list = query_indices_base(term, tmp_indices)
        result_list.extend(tmp_docID_list)
    return sorted(set(result_list))   # 直接集合操作复杂度可能高

def write_file(results,output_path):
    """
    将list结果写入输出文件
    :param results: 结果
    :param output_path: 输出文件
    :return: 0
    """
    print ("Writing the Data to the Output Path: ", output_path)
    with open(output_path,'w') as f:
        results_len = len(results)
        num = 0
        for result in results:
            if len(result) == 0:
                print ("ERROR!")                           # 有空行就报错
            elif len(result) > 0:
                num += 1
                result = [str(i) for i in result]
                if num == results_len:
                    line = ' '.join(result)
                else:
                    line = ' '.join(result) + '\n'
                f.write(line)
    print ("    Data Writed!")
    return 0

if __name__ == '__main__':
    start_time = time.time()
    # 输入输出文件路径
    inverted_indices_path = './2_standard_inverted_indices.txt'
    permuterm_indices_path = './2_standard.txt'
    queries_path = './3.txt'
    output_path = './4_generated.txt'

    # 加载倒排索引、轮排对照表和查询
    indices_list = load_inverted_indices(inverted_indices_path)
    permuterm_indices = load_permuterm_indices(permuterm_indices_path)
    queries = load_queries(queries_path)
    print (permuterm_indices[-100:])

    # 对每一条查询进行处理
    query_num = 0
    all_result = []
    print ("Querying ...")
    for query in queries:
        if (query_num + 1) % 500 == 0:
            print ("    Processing Query Num: ", query_num+1)
        tmp_result = query_indices(query,permuterm_indices,indices_list)  # 文档id list
        all_result.append(tmp_result)
        query_num += 1
    print ("    Queries Processed!")

    # 输出到output文件
    write_file(all_result,output_path)
    end_time = time.time()
    print ("Time Used: ", end_time - start_time)