import websockets
import json
import os
from dotenv import load_dotenv
from utils import motor_update

load_dotenv()

async def handler(websocket):
    """
    Handle incoming WebSocket connections and motor control messages
    """
    client_ip = websocket.remote_address[0]
    print(f"Client connected from {client_ip}")

    try:
        async for message in websocket:
            
            try:
                data = json.loads(message)
                motor_update(data)

                # Send success response
                response = {
                    "status": "success",
                    "message": "Motors updated"
                    # "received_signals": message
                }
                
            except json.JSONDecodeError:
                response = {
                    "status": "error",
                    "message": "Invalid JSON format"
                }
            
            # Send response back to client
            await websocket.send(json.dumps(response))
            print(f"Sent response: {response}")
            
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    except Exception as e:
        print(f"Error in handler: {e}")

async def start_server():
    """
    Main entry point to start the WebSocket server
    """
    server = await websockets.serve(
        handler,
        os.getenv("WEBSOCKET_HOST"),  # Listen on all interfaces
        os.getenv("WEBSOCKET_PORT")        # Port from your .env
    )
    print(f"Motor control WebSocket server started on ws://{os.getenv('WEBSOCKET_HOST')}:{os.getenv('WEBSOCKET_PORT')}")
    await server.wait_closed()