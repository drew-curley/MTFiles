from collections import defaultdict

# Your string of language-region entries
with open("../constants/all_supported_NLLB_languages.txt") as file:
    lang_string = file.read()

# Split the string by commas and remove extra whitespace
lang_entries = [entry.strip() for entry in lang_string.split(",")]

# Dictionary to store language codes and their corresponding entries
lang_dict = defaultdict(list)

# Loop through the entries and group by language code (first 3 characters)
for entry in lang_entries:
    lang_code = entry[:3]
    lang_dict[lang_code].append(entry)

# Find and print entries with duplicate language codes
duplicates = {code: entries for code, entries in lang_dict.items() if len(entries) > 1}

# Output the results
for code, entries in duplicates.items():
    print(f"Language code '{code}' has the following entries: {entries}")
