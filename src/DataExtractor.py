import os

import requests
from dotenv import load_dotenv

load_dotenv()


class DataExtractor:

    def __init__(self, token: str = None, url: str = None):
        self.url = url or os.getenv("BASE_URL")
        self.token = token or os.getenv("TOKEN")
        self.movies = []

        if not self.url:
            raise ValueError(
                "TMDB API URL must be provided or set in environment variables."
            )

        if not self.token:
            raise ValueError(
                "TMDB API Read Access Token must be provided "
                "or set in environment variables."
            )

    def extract_data(self, page : int = 100):

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

        param = {
            "page" : page
        }
        try:
            response = requests.get(
                self.url,
                headers=headers,
                params=param,
                timeout=10
            )

            response.raise_for_status()
            data = response.json()

            movies = data.get("results", [])

            self.movies.extend(movies)

            return movies

        except requests.exceptions.HTTPError as http_err:
            print(
                f"HTTP Error occurred: {http_err} "
                f"- Status Code: {response.status_code}"
            )
            return {}

        except requests.exceptions.RequestException as err:
            print(f"Request Error occurred: {err}")
            return {}

    def extract_all_data(self, page: int = 100):

        self.extract_data(100)
        all_results = []

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

        for movie in self.movies:

            movie_id = movie["id"]

            url = f"https://api.themoviedb.org/3/movie/{movie_id}?append_to_response=keywords"

            try:
                response = requests.get(
                    url,
                    headers=headers,
                    timeout=10
                )

                response.raise_for_status()

                movie_data = response.json()

                all_results.append(movie_data)

            except requests.exceptions.RequestException as error:
                print(
                    f"Error extracting movie {movie_id}: {error}"
                )

        return all_results