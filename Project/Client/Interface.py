import socket
import struct
def TestConnection(HOST,PORT):
    operator="TestConnection"
    operator=bytes(operator,'utf-8')
    message=struct.pack('>I', len(operator)) + operator
    try:
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
            s.connect((HOST,PORT))#建立连接
            s.sendall(message)#发送数据给服务端
            print(s.recv(1))
    except Exception as ex:
        return str(ex)
    return "True"