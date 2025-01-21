from redis import Redis
from dotenv import load_dotenv
load_dotenv()
import os

def main():
    redis_client = Redis.from_url(os.getenv('REDIS_URL'))
    redis_client.set('hello2', 'world')
    print(redis_client.get('hello2'))




if __name__ == '__main__':
    main()