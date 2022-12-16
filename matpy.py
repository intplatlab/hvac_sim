import socket
import textwrap

def soc(in1, in2, in3, in5, in4, in6, in7):
    HOST = '127.0.0.1'
    PORT = 9999

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))

    client_socket.sendall('{}, {}, {}, {}, {}, {}, {}'.format(in1, in2, in3, in5, in4, in6, in7).encode())
    data = client_socket.recv(1024).decode()
    client_socket.close()
    data = data.split(',')

    out = int(data[0])
    if out == 0:
        out = -1
    else:
        out = 1
    next_t = float(data[1])

    result = out * next_t
    result = float(result)
  
    return result
