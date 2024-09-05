import re
import requests
import json
import os

with open("./constants/all_supported_NLLB_languages.txt") as file:
    all_supported_NLLB_languages = file.read()

url = "https://api.bibleineverylanguage.org/v1/graphql"
json_file_path = "./constants/wa-language-metadata.json"

query = """
query MyQuery {
  wa_language_metadata(where: {is_gateway: {_eq: true}}) {
    is_gateway
    language {
      english_name
      iso6393
    }
  }
}
"""

response = requests.post(url, json={'query': query})

if response.status_code == 200:
    data = response.json()
    wa_language_metadata = data['data']['wa_language_metadata']

    for gateway_language in wa_language_metadata:
        regex = fr"{gateway_language['language']['iso6393']}_\w+"
        matches = re.findall(regex, all_supported_NLLB_languages)
        gateway_language['language']['supported_nllb_variants'] = matches


        os.makedirs(os.path.dirname(json_file_path), exist_ok=True)

        with open(json_file_path, "w", encoding="utf-8") as json_file:
            json.dump({'wa_language_metadata': wa_language_metadata}, json_file, ensure_ascii=False, indent=4)

else:
    # Handle errors
    print(f"Query failed with status code {response.status_code}")
    print(response.text)







