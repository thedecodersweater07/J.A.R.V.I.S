import asyncio
from ml.data_collector import MLDataCollector

async def main():
    # Example sources (add your own)
    sources = [
        "https://example.com/machine-learning",
        "https://example.com/artificial-intelligence",
        "https://example.com/data-science"
    ]
    
    collector = MLDataCollector()
    
    # Collect and process data
    df = await collector.collect_and_process(sources, "ml_training_data")
    
    # Merge all existing datasets
    merged_df = collector.merge_datasets()
    print(f"Total collected samples: {len(merged_df)}")

if __name__ == "__main__":
    asyncio.run(main())
