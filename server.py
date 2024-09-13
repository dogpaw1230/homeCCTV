import asyncio
import websockets

async def handler(websocket, path):
    try:
        async for message in websocket:
            print(f"Received message of length {len(message)}")
            # 클라이언트가 보내는 메시지를 그대로 응답 (에코 서버의 예)
            await websocket.send(message)
    except websockets.ConnectionClosedError as e:
        print(f"Connection closed with error: {e}")

# 웹소켓 서버 설정 (메시지 크기 16MB로 설정)
start_server = websockets.serve(handler, "localhost", 8081, max_size=2**24)  # 16MB

async def main():
    server = await start_server
    async with server:
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
