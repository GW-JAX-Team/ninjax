"""
Argument parser, adapted from bilby_pipe

TODO: IMPROVEMENTS NEEDED - This parser is functional but has several areas for improvement

      POTENTIAL ENHANCEMENTS:
      1. Add JSON schema validation for config files
      2. Better error messages when parsing fails (show line numbers, context)
      3. Support for environment variable substitution
      4. Support for config file includes/imports
      5. Type validation (detect int vs float vs bool)
      6. Add support for YAML config files (more human-readable)

      TESTING NEEDED:
      - Test multi-line dictionaries with nested structures
      - Test edge cases (empty files, malformed input)
      - Test comments with special characters
      - Add unit tests for parser edge cases
"""

import re
import json
from collections import OrderedDict

import ninjax.pipes.pipe_utils as utils
from ninjax.pipes.pipe_utils import logger, DuplicateErrorDict

class ConfigParser(object):
    """Parse .ini and .json configuration files

    TODO: Consider adding:
          - Config validation against schema
          - Better type inference (currently everything is string)
          - Support for config inheritance/composition
    """
    
    def parse(self, filename: str) -> dict:
        if filename.endswith(".json"):
            return self.parse_json_file(filename)
        elif filename.endswith(".ini"):
            return self.parse_ini_file(filename)
        else:
            raise ValueError("Unrecognized file extension")
        
    def parse_json_file(self, filename: str) -> dict:
        with open(filename, "r") as f:
            return json.load(f)
    
    def parse_ini_file(self, filename: str) -> OrderedDict:
        """Parses the keys + values from a config file. Credit to bilby_pipe for this method."""

        # Open the file to get the stream
        try:
            with open(filename, "r") as f:
                stream = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"File {filename} not found")

        # Pre-process lines to put multi-line dicts on single linesj
        lines = list(stream)  # Turn into a list
        
        lines = [line.strip(" ") for line in lines]  # Strip trailing white space
        first_chars = [line[0] for line in lines]  # Pull out the first character

        ii = 0
        lines_repacked = []
        while ii < len(lines):
            if first_chars[ii] == "#":
                lines_repacked.append(lines[ii])
                ii += 1
            else:
                # Find the next comment
                jj = ii + 1
                while jj < len(first_chars) and first_chars[jj] != "#":
                    jj += 1

                int_lines = "".join(lines[ii:jj])  # Form single string
                int_lines = int_lines.replace(",\n#", ", \n#")  #
                int_lines = int_lines.replace(
                    ",\n", ", "
                )  # Multiline args on single lines
                int_lines = int_lines.replace(
                    "\n}\n", "}\n"
                )  # Trailing } on single lines
                int_lines = int_lines.split("\n")
                lines_repacked += int_lines
                ii = jj

        # items is where we store the key-value pairs read in from the config
        # we use a DuplicateErrorDict so that an error is raised on duplicate
        # entries (e.g., if the user has two identical keys)
        items = DuplicateErrorDict()
        numbers = dict()
        comments = dict()
        inline_comments = dict()
        for ii, line in enumerate(lines_repacked):
            line = line.strip()
            if not line:
                continue
            if line[0] in ["#", ";", "["] or line.startswith("---"):
                comments[ii] = line
                continue
            if len(line.split("#")) > 1:
                inline_comments[ii] = "  #" + "#".join(line.split("#")[1:])
                line = line.split("#")[0]
            white_space = "\\s*"
            key = r"(?P<key>[^:=;#\s]+?)"
            value = white_space + r"[:=\s]" + white_space + "(?P<value>.+?)"
            comment = white_space + "(?P<comment>\\s[;#].*)?"

            key_only_match = re.match("^" + key + comment + "$", line)
            if key_only_match:
                key = key_only_match.group("key")#.replace("_", "-")
                items[key] = "true"
                numbers[key] = ii
                continue

            key_value_match = re.match("^" + key + value + comment + "$", line)
            if key_value_match:
                key = key_value_match.group("key")#.replace("_", "-")
                value = key_value_match.group("value")

                if value.startswith("[") and value.endswith("]"):
                    # handle special case of lists
                    value = [elem.strip() for elem in value[1:-1].split(",")]

                items[key] = value
                numbers[key] = ii
                continue

            raise ValueError(
                f"Unexpected line {ii} in {getattr(stream, 'name', 'stream')}: {line}"
            )

        items = self.reconstruct_multiline_dictionary(items)
        # TODO: do we want to have those other returns?
        # return items, numbers, comments, inline_comments
        return items

    def reconstruct_multiline_dictionary(self, items: dict) -> dict:
        # Convert items into a dictionary to avoid duplicate-line errors
        items = dict(items)
        keys = list(items.keys())
        vals = list(items.values())
        for ii, val in enumerate(vals):
            if "{" in val and "}" not in val:
                sub_ii = 1
                sub_dict_vals = []
                if val != "{":
                    sub_dict_vals.append(val.rstrip("{"))
                while True:
                    next_line = f"{keys[ii + sub_ii]}: {vals[ii + sub_ii]}"
                    items.pop(keys[ii + sub_ii])
                    if "}" not in next_line:
                        if "{" in next_line:
                            raise ValueError("Unable to pass multi-line config file")
                        sub_dict_vals.append(next_line)
                        sub_ii += 1
                        continue
                    elif next_line == "}: true":
                        sub_dict_vals.append("}")
                        break
                    else:
                        sub_dict_vals.append(next_line)
                        break
                sub_dict_vals_with_comma = [
                    vv.rstrip(",").lstrip(",") for vv in sub_dict_vals
                ]
                items[keys[ii]] = "{" + (", ".join(sub_dict_vals_with_comma)).lstrip(
                    "{"
                )
        return items