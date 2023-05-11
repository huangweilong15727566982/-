import mysql.connector
import cv2
import numpy as np
import threading
import struct
# 创建连接   
mydb = mysql.connector.connect(
  host="192.168.222.128",
  user="root",
  password="Itheima66^",
  database="Object",
  charset="utf8"
)
# 创建游标
mycursor = mydb.cursor()
# shared_data=[]#共享数据须互斥访问
lock=threading.Lock()#创建一个锁对象
# def thread_GetTask():#定义线程函数 获取标注任务
#   global shared_data,lock
#   lock.acquire()#获得锁 未获得锁会被卡住
#   #获取一个任务集并从数据库删除
#   cursor=mydb.cursor()
#   query="SELECT * FROM picture"
#   cursor.execute(query)
#   result = cursor.fetchall()
#   shared_data = result  
#   # 释放锁
#   lock.release()
# 执行 SQL 查询
def addPicture(image,string,table):#添加一张图片 指明向哪个表添加图像
  #a=f"INSERT INTO picture(image,id) VALUES({image},{id});"
  lock.acquire()#获得锁 未获得锁会被卡住
  sql = f"INSERT INTO {table} (image,label) VALUES (%s,%s)"
  values=(image,string)
  mycursor.execute(sql, values)
  mydb.commit()#提交事务
  lock.release()
  print("sql_ok!")
#提取出数据库中的图片数据  
def getPicture(table):#指明获取哪个表的数据集
  sql=f"select image,label from {table}"
  mycursor.execute(sql)
  result= mycursor.fetchall()
  faces=[]
  target=[]
  for row in result:
    # img_array=np.frombuffer(row[0],np.uint8)
    # img=cv2.imdecode(img_array,cv2.IMREAD_GRAYSCALE)
    faces.append(row[0])#img
    target.append(row[1])#label 
  # 转换成训练需要的数组
  # faces=np.asarray(faces)
  # target=np.asarray(target) 
  return faces,target
def showPic(table):
  sql=f"select * from {table};"
  mycursor.execute(sql)
  # 获取结果
  result = mycursor.fetchall()
  # 输出结果
  for row in result:
    print(row)
  print("1111")

#添加任务包 
#在录入任务包时需指定为识别哪个物体所对应的表  
def addTaskSet(table,taskSet):   #taskSets存放taskSet 
    for item in taskSet:
      lock.acquire()#获得锁 未获得锁会被卡住
      sql=f"INSERT INTO TASKS (image,table_name) VALUES(%s,%s);"
      values=(item,table)
      mycursor.execute(sql,values)
      mydb.commit()#提交事务
      print("sql_ok!")
      lock.release()#获得锁 未获得锁会被卡住
#获取任务包
def AcquireTask():
  #每次最多获取20条任务进行标注
    sql="SELECT image,table_name,unique_id FROM TASKS LIMIT 5;"
    mycursor.execute(sql)
    result=mycursor.fetchall()
    message=None
    #格式 图片长度，图片bits 表长 表名 ...end 
    for item in result:
      if message:
        message+=struct.pack('>I',len(item[0]))+item[0]
      else:
        message=struct.pack('>I',len(item[0]))+item[0]
      string=bytes(item[1],'utf-8')
      message+=struct.pack('>I',len(string))+string
    if message:
      message+=struct.pack('>I',0xffffffff)
    else: message=struct.pack('>I',0xffffffff)
    sql = "DELETE FROM TASKS WHERE unique_id IN "
    params = tuple([row[2] for row in result])
    if len(params)==0:return message
    if len(params)==1:
      sql+=f"({params[0]});"
    else: sql += str(params) + ";"
    print(sql)
    mycursor.execute(sql)
    mydb.commit()
    return message
      
  # #获取最大的id
  # sql="SELECT MAX(id) FROM TaskSet;"
  # mycursor.execute(sql)
  # result=mycursor.fetchall()
  # # id=0
  # if result[0][0]!=None:id=int(result[0][0])
  # else: return None
  # sql=f"SELECT * FROM TaskSet where id={id}"
  # mycursor.execute(sql)
  # result=mycursor.fetchall()
  # sql=f"DELETE FROM TaskSet WHERE id = {id};"
  # mycursor.execute(sql)
  # mydb.commit()#提交事务
  # message=None
  # for row in result:
  #   message+=struct.pack('>I', len(row[1])) + row[1] 
  # return message
#提交数据集
def CommitDataSet(DataSet):
  for id,img in DataSet:
    addPicture(id,img)
def AcquireDataSet(sock,table):#获取相应表中的数据集
  sql=f"select image,labels from {table}"
  mycursor.execute(sql)
  row = mycursor.fetchone() #每次获取一行数据
  while row is not None:
    # 处理当前行数据
    message=struct.pack('>I',len(row[0]))+row[0]
    string=bytes(row[1],'utf-8')
    message+=struct.pack('>I',len(string))+string
    sock.sendall(message)
    sock.recv(1)#接受确认号
    row = mycursor.fetchone()  # 获取下一行数据
  sock.sendall(struct.pack('>I',0xffffffff))
  # sock.recv(1)#接受确认号
      
#AcquireDataSet("Gensin")
  # image,id=getPicture()
  # #打包成消息发送给客服端
  # message=None
  # print("Len(image,id)",len(image),len(id))
  # for img,Id in zip(image,id):
  #   if message==None:
  #     message=struct.pack('>I',len(img))
  #   else: message+=struct.pack('>I',len(img))
  #   message+=struct.pack('>I',Id)
  #   message+=img
  # message+=struct.pack('>I',0xffffffff)
  # return message

#添加一个任务
def AddOnePic(img,string,table):
  sql = f"INSERT INTO {table} (image,labels) VALUES (%s,%s)"
  values=(img,string)
  mycursor.execute(sql, values)
  mydb.commit()#提交事务
def GetTableInfo():
  message=None
  sql="select * from TABLE_NAME"
  mycursor.execute(sql)
  result=mycursor.fetchall()
  print(result)
  for item in result:
    string=""
    string+=str(item[0])
    string+=" "
    string+=str(item[1])
    string+=" "
    string+=str(item[2])
    string=bytes(string,'utf-8')
    if message:
      message+=struct.pack('>I',len(string))+string
    else:
      message=struct.pack('>I',len(string))+string
  message+=struct.pack('>I',0xffffffff)

  print(result)
  return message
#GetTableInfo()

#mycursor.execute(r"CREATE TABLE names ( id INT,name VARCHAR(50));")

# 获取结果
#result = mycursor.fetchall()

# # 输出结果
# for row in result:
#   print(row)
# print("1111")
