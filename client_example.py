"""
RIC Pipeline — client_example.py
Simple example showing how to use the pipeline directly (no server).
"""
import asyncio
from ric.pipeline import RICPipeline

async def main():
    pipeline = RICPipeline()
    await pipeline.init()

    queries = [
        "What is photosynthesis?",          # → local
        "Write a Python web scraper.",       # → cloud
        "What is photosynthesis?",           # → cache (instant!)
    ]

    for q in queries:
        result = await pipeline.run(q)
        print(f"\n{'='*60}")
        print(f"Query    : {q}")
        print(f"Dest     : {result.destination.upper()}")
        print(f"Latency  : {result.latency_ms:.0f} ms")
        print(f"Tokens   : {result.total_tokens}")
        print(f"Cached?  : {result.from_cache}")
        print(f"Response : {result.response[:120]}...")

    stats = await pipeline.get_stats()
    print(f"\n{'='*60}")
    print(f"Cache hit rate: {stats['cache']['hit_rate_pct']}%")
    print(f"Tokens saved : {stats['cache']['total_saved_tokens']}")

    await pipeline.close()

if __name__ == "__main__":
    asyncio.run(main())
