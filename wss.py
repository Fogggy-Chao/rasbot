import websockets
import json
import os
import asyncio
from dotenv import load_dotenv
from robot_interface import execute_tool_call
import functools
import asyncio

load_dotenv()

# Global variable to hold robot contexts, will be set by main.py
# This is one way to make it accessible; another is passing via functools.partial
# For now, let's aim to pass it directly via the server setup.

async def handler(websocket, robot_contexts):
    """
    Handle incoming WebSocket connections and tool call messages.
    """
    client_ip = websocket.remote_address[0]
    print(f"Client connected from {client_ip}")

    try:
        async for message in websocket:
            response = None # Initialize response
            try:
                data = json.loads(message)
                tool_name = data.get("name")
                arguments = data.get("arguments")

                if not tool_name:
                    response = {
                        "status": "error",
                        "message": "Missing 'name' (tool name) in request."
                    }
                else:
                    # Call the central tool executor
                    # robot_contexts will be passed from start_server
                    tool_result = await asyncio.to_thread(
                        execute_tool_call, tool_name, arguments, robot_contexts
                    )
                    # The AI expects the direct result of the tool call (or a DONE:/ERROR: message)
                    # Our execute_tool_call should return a dictionary that can be directly sent
                    # or adapted if the AI expects a specific top-level structure for tool results.
                    # Based on the examples, the AI expects the direct JSON output of the tool.
                    response = tool_result
                
            except json.JSONDecodeError:
                response = {
                    "status": "error",
                    "message": "Invalid JSON format"
                }
            except Exception as e:
                print(f"Error processing tool call: {e}")
                response = {
                    "status": "error",
                    "message": f"Error processing request: {str(e)}"
                }
            
            # Send response back to client
            if response:
                await websocket.send(json.dumps(response))
                print(f"Sent response: {json.dumps(response)}")
            
    except websockets.exceptions.ConnectionClosed:
        print(f"Client {client_ip} disconnected")
    except Exception as e:
        print(f"Error in WebSocket handler for {client_ip}: {e}")

async def start_server(robot_contexts):
    """
    Main entry point to start the WebSocket server.
    """
    # Use functools.partial to pass robot_contexts to the handler
    bound_handler = functools.partial(handler, robot_contexts=robot_contexts)
    
    server_host = os.getenv("WEBSOCKET_HOST", "0.0.0.0")
    server_port = int(os.getenv("WEBSOCKET_PORT", 3001))

    async with websockets.serve(bound_handler, server_host, server_port) as server:
        print(f"Robot control WebSocket server started on ws://{server_host}:{server_port}")
        await server.wait_closed()