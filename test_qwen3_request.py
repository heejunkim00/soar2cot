#!/usr/bin/env python
"""Test Qwen3 server with a simple request"""
import asyncio
import time
from openai import AsyncOpenAI

async def test_qwen3():
    client = AsyncOpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
        timeout=300
    )

    print("Sending test request to Qwen3...")
    start = time.time()

    try:
        response = await client.chat.completions.create(
            model="Qwen/Qwen3-32B",
            messages=[{"role": "user", "content": "Hello, say hi back to me!"}],
            max_tokens=50
        )
        duration = time.time() - start
        print(f"✓ Success! Duration: {duration:.2f}s")
        print(f"Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        duration = time.time() - start
        print(f"✗ Failed after {duration:.2f}s")
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_qwen3())
