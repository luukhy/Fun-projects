import socket
from threading import Thread

class Server:
    clients = []

    def __init__(self, HOST, PORT):
        self.m_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.m_socket.bind((HOST, PORT))
        self.m_socket.listen(5)
        print('Server waiting for connection...')

    def listen(self):
        while True:
            client_socket, address = self.m_socket.accept()
            print(f'Connection from: {address}')

            client_name = client_socket.recv(1024).decode()
            client = {'client_name': client_name, 'client_socket': client_socket}
            self.clients.append(client)

            self.broadcast_message(client_name, f'{client_name} has joined the chat!')
            Thread(target=self.handle_client, args=(client,)).start()

    def handle_client(self, client):
        client_name = client['client_name']
        client_socket = client['client_socket']
        
        while True:
            client_message = client_socket.recv(1024).decode()

            if client_message.strip() == client_name + ": logout" or not client_message.strip():
                self.broadcast_message(client_name, f'{client_name} has left the chat!')
                self.clients.remove(client)
                client_socket.close()
                break
            else:
                self.broadcast_message(client_name, client_message)
    
    def broadcast_message(self, sender_name, message):
        for client in self.clients:
            client_socket   = client['client_socket']
            client_name     = client['client_name']
            if client_name != sender_name:
                client_socket.send(message.encode())

if __name__ == '__main__':
    server = Server('127.0.0.1', 7632)
    server.listen()