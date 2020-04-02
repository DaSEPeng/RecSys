***********************************
实验任务：构建轮排索引
Author: 李鹏
StuID: 10175501102
Date: 2020/4/1
***********************************

python 文件名
    - construct_index.py

实验环境：
    - Python 3.7
    - numpy 1.15.4
    - PyCharm

数据处理过程简述：
    本次实验在过程中假设了文档数目非常多且词表数非常大的情况，对几个步骤进行了优化，因此与提供的参考步骤可能略有不同
    - 读取文件
      使用f.readline()逐行读入文件，在逐行读入的同时使用python的字符串处理函数lower()将所有的大写转化成了小写
    - 构建轮排索引
      遍历整个数据集，统计词频保存到{term:term_frequency}的字典A中
      使用numpy的argsort()对term_frequency进行排序，得到top100的index值
      使用numpy的argsort()对term进行排序，得到对应的index值
      遍历排序好的term的index，如果不在top100 index的列表中，就读取对应的term
      利用rotate_base(str,n)和rotate(str)函数得到当前term的轮排变化List
      按照实验要求将结果组织成相应的格式
    - 输出文件
      将文件输出到output的路径

运行（假设命令行，PyCharm环境大致相同）：
    - 运行 python3 construct_index.py
    - 便会得到2_generated.txt文件
    - 运行 python3 compare_index.py检查输出结果与标准结果是否相同

备注：
    - 因为对空格的处理，以及去除TOP100的词频过程不稳定，因为输出文件和标准文件略有不同
    （但是查过python的sort函数是稳定的，具体原因可能要以后再排查）
    - 本次实验假设了实际生产环境中数据量很大的问题，并对数据处理过程进行了一定的优化，但是优化的准确性还有有待讨论
    - 处理文件的时候需要注意最后一行有没有换行符，要不要输出换行符