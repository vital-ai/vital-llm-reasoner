import json
import logging
import requests
from serpapi import GoogleSearch
from vital_llm_reasoner.config.reasoner_config import ReasonerConfig


def main():
    logging.basicConfig(level=logging.INFO)

    config_file_path = "../reasoner_config.yaml"
    reasoner_config = ReasonerConfig(config_file_path)

    google_search_key = reasoner_config.google_search_key

    query = "when is Jimmy Carter's Birthday"

    params = {
        "engine": "google",
        "q": query,
        "api_key": google_search_key
    }

    try:
        search = GoogleSearch(params)

        if search.get_response().status_code == 200:
            results = search.get_dict()

            pretty_results = json.dumps(results, indent=4)

            print(pretty_results)

            organic_results = results["organic_results"]

            print(organic_results )

        else:
            print(f"Error: {search.get_response().status_code}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
