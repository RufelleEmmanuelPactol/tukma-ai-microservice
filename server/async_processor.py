import PyPDF2
import redis
from redis import Redis
from dotenv import load_dotenv
import os
import threading
from typing import List
import logging
from contextlib import contextmanager

from ai.similarity_scorer import SimilarityScorer

load_dotenv()



@contextmanager
def pdf_reader(file):
    try:
        reader = PyPDF2.PdfReader(file)
        yield reader
    except Exception as e:
        print(f"Error reading PDF: {e}")
        raise


def process_to_pdf(file) -> str:
    with pdf_reader(file) as reader:
        return "\n".join(page.extract_text() for page in reader.pages)


class ResumeProcessor:
    def __init__(self):
        self.redis_client: redis.Redis = Redis.from_url(
            os.getenv('REDIS_URL'),
            decode_responses=True
        )
        self.scorer = SimilarityScorer()
        threading.Thread(target=self.run_delays, daemon=True).start()
        print('Resolver thread now running')

    def submit(self, file, keywords: List[str], item_hash: str) -> None:
        thread = threading.Thread(
            target=self.run_process,
            args=(file, keywords, item_hash),
            daemon=True
        )
        thread.start()
        print(f"Processing thread started for hash: {item_hash}")

    def run_process(self, file, keywords: List[str], item_hash: str) -> None:
        try:
            text = file
            pipe = self.redis_client.pipeline()

            # Store temporary data
            pipe.setex(f"volatile_{item_hash}", 360000, text)  # 1 hour expiry
            pipe.rpush(f"volt_keys_{item_hash}", *keywords)
            pipe.execute()

            # Calculate score
            score = self.scorer.calculate_relevance_scores(text, keywords)

            # Update final results
            pipe = self.redis_client.pipeline()
            pipe.set(item_hash, score)
            pipe.delete(f"volatile_{item_hash}")
            pipe.delete(f"volt_keys_{item_hash}")
            print('deleting any overdue keys')
            pipe.execute()

        except Exception as e:
            print(f"Error processing resume {item_hash}: {e}")
            raise e

    def run_delays(self) -> None:
        while True:
            try:
                for key in self.redis_client.scan_iter("volatile_*"):
                    try:
                        base_hash = key.removeprefix("volatile_")
                        keywords = self.redis_client.lrange(f'volt_keys_{base_hash}', 0, -1)
                        text = self.redis_client.get(key)

                        if text and keywords:
                            score = self.scorer.calculate_relevance_scores(text, keywords)
                            self.redis_client.set(base_hash, str(score))

                        # Cleanup
                        self.redis_client.delete(key)
                        self.redis_client.delete(f'volt_keys_{base_hash}')

                    except Exception as e:

                        raise e

            except Exception as e:
                raise e

            # Sleep between scans
            threading.Event().wait(60)

    def get_result(self, item_hash: str) -> str:
        try:
            return self.redis_client.get(item_hash)



        except Exception as e:
            print(f"Error retrieving result for {item_hash}: {e}")
            raise

    def check_item_status(self, item_hash: str) -> str:
        if self.redis_client.exists(item_hash):
            return "Item exists, result ready."
        if self.redis_client.exists(f'volatile_{item_hash}'):
            return "Item exists, result processing."
        return None