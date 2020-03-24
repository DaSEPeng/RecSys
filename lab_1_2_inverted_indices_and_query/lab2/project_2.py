"""

推荐系统实验2：倒排索引查询

@Author：李鹏 (ruhao9805@163.com)
@Date: 2020/3/24
"""

import time

def load_indices(file_path):
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
        result_docID_list = docID_list[index]
    except:
        result_docID_list = []
    return result_docID_list

def get_list_intersection(list_a,list_b):
    """
    得到两个列表的交集
    假设列表内已经升序排序，利用两个指针采用类似归并排序的方式得到列表的交集
    :param list_a: 第一个列表
    :param list_b: 第二个列表
    :return: 两个列表的交集，保持升序
    """
    result = []
    if len(list_a)==0 or len(list_b) == 0:
        return result
    else:
        list_a_len = len(list_a)
        list_b_len = len(list_b)
        ptr_a = 0                           # list a 的指针
        ptr_b = 0                           # list b 的指针
        while True:
            # 终点处理
            if ptr_a == list_a_len or ptr_b == list_b_len:
                break
            # 迭代求并集
            tmp_a_str = list_a[ptr_a]
            tmp_b_str = list_b[ptr_b]
            tmp_a_int = eval(tmp_a_str)
            tmp_b_int = eval(tmp_b_str)
            if tmp_a_int == tmp_b_int:
                result.append(tmp_a_str)
                ptr_a += 1
                ptr_b += 1
            elif tmp_a_int < tmp_b_int:
                ptr_a += 1
            elif tmp_b_int < tmp_a_int:
                ptr_b += 1

    return result

def get_list_union(list_a,list_b):
    """
    得到两个列表的并集
    假设列表内已经升序排序，利用两个指针采用类似归并排序的方式得到列表的并集
    :param list_a: 第一个列表
    :param list_b: 第二个列表
    :return: 两个列表的并集，保持升序
    """
    result = []
    if len(list_a)==0 or len(list_b) == 0:
        return result
    else:
        list_a_len = len(list_a)
        list_b_len = len(list_b)
        ptr_a = 0                           # list a 的指针
        ptr_b = 0                           # list b 的指针
        while True:
            # 终点处理
            if ptr_a == list_a_len and ptr_b == list_b_len:
                break
            if ptr_a == list_a_len:
                result.extend(list_b[ptr_b:])
                break
            if ptr_b == list_b_len:
                result.extend(list_a[ptr_a:])
                break
            # 迭代求并集
            tmp_a_str = list_a[ptr_a]
            tmp_b_str = list_b[ptr_b]
            tmp_a_int = eval(tmp_a_str)
            tmp_b_int = eval(tmp_b_str)
            if tmp_a_int == tmp_b_int:
                result.append(tmp_a_str)
                ptr_a += 1
                ptr_b += 1
            elif tmp_a_int < tmp_b_int:
                result.append(tmp_a_str)
                result.append(tmp_b_str)
                ptr_a += 1
                ptr_b += 1
            elif tmp_a_int > tmp_b_int:
                result.append(tmp_b_str)
                result.append(tmp_a_str)
                ptr_a += 1
                ptr_b += 1
    return result

def query_indices(tmp_query,tmp_indices):
    """
    处理一个query
    :param tmp_query: 当前的query
    :param tmp_indices: 倒排索引表
    :return: 查询结果
    """
    tmp_result = ''
    tmp_query_splited = tmp_query.split(' ')
    if len(tmp_query_splited) == 1:
        tmp_result = query_indices_base(tmp_query_splited[0],tmp_indices)
    elif len(tmp_query_splited) == 3:
        query_a = tmp_query_splited[0]
        query_b = tmp_query_splited[2]
        op = tmp_query_splited[1]
        docID_a = query_indices_base(query_a,tmp_indices)
        docID_b = query_indices_base(query_b,tmp_indices)
        if op == 'and':
            tmp_result = get_list_intersection(docID_a,docID_b)
        elif op == 'or':
            tmp_result = get_list_union(docID_a,docID_b)
    return tmp_result

def write_file(results,output_path):
    """
    将list结果写入输出文件
    :param results: 结果
    :param output_path: 输出文件
    :return: 0
    """
    print ("Writing the Data to the Output Path: ", output_path)
    with open(output_path,'w') as f:
        for result in results:
            if len(result) == 0:
                f.write('\n')
            elif len(result) > 0:
                line = ' '.join(result) + '\n'
                f.write(line)
    print ("    Data Writed!")
    return 0

if __name__ == '__main__':
    start_time = time.time()
    # 输入输出文件路径
    inverted_indices_path = './2_standard.txt'
    queries_path = './3.txt'
    output_path = './4_generated.txt'

    # 加载倒排索引和查询
    indices_list = load_indices(inverted_indices_path)
    queries = load_queries(queries_path)

    # 对每一条查询进行处理
    query_num = 0
    all_result = []
    print ("Querying ...")
    for query in queries:
        if (query_num + 1) % 100 == 0:
            print ("    Processing Query Num: ", query_num+1)
        tmp_result = query_indices(query,indices_list)
        all_result.append(tmp_result)
        query_num += 1
    print ("    Queries Processed!")

    # 输出到output文件
    write_file(all_result,output_path)
    end_time = time.time()
    print ("Time Used: ", end_time - start_time)