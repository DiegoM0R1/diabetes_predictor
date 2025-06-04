# core/consumers.py
import json
from channels.generic.websocket import WebsocketConsumer
from asgiref.sync import async_to_sync

class TrainingConsumer(WebsocketConsumer):
    def connect(self):
        self.room_name = "training_progress"
        self.room_group_name = "training_progress"

        # Join room group
        async_to_sync(self.channel_layer.group_add)(
            self.room_group_name, self.channel_name
        )

        self.accept()
        self.send(text_data=json.dumps({
            'message': 'Conectado al canal de progreso de entrenamiento.',
            'stage': 0
        }))

    def disconnect(self, close_code):
        # Leave room group
        async_to_sync(self.channel_layer.group_discard)(
            self.room_group_name, self.channel_name
        )

    # Receive message from WebSocket
    def receive(self, text_data):
        # Este consumidor solo envía mensajes, no recibe comandos del cliente en esta demo.
        pass

    # Receive message from room group
    def send_message(self, event):
        message = event['message']
        stage = event.get('stage', 0)
        # Send message to WebSocket
        self.send(text_data=json.dumps({
            'message': message,
            'stage': stage
        }))
    
    def send_progress(self, event):
        # Para enviar métricas por época (simuladas)
        message = event['message']
        epoch = event['epoch']
        total_epochs = event['total_epochs']
        train_error = event['train_error']
        valid_error = event['valid_error']
        stage = event['stage']

        self.send(text_data=json.dumps({
            'type': 'progress_update',
            'message': message,
            'epoch': epoch,
            'total_epochs': total_epochs,
            'train_error': train_error,
            'valid_error': valid_error,
            'stage': stage
        }))

    def redirect_to_dashboard(self, event):
        url = event['url']
        self.send(text_data=json.dumps({
            'type': 'redirect',
            'url': url
        }))