#! /usr/bin/env python3


def getPars(filename):
    return readBook(filename).splitlines()
    

def readBook(filename):
    """
    Returns the text of a file.
    """
    with open(filename, "r") as f:
        text = f.read()

    return dealign(text)


def dealign(text):
    """
    Original text is limited to 72 characters per line, paragraphs separated
    by a blank line. This function merges every paragraph to one line.
    
    Args:
        text (str): Text with limited line size.
    
    Returns:
        str: Text with every paragraph on a single line.
    """
    original_lines = text.splitlines()
    paragraphs = []
    i = 0
    while i < len(original_lines):
        paragraph_lines = []
        while True:
            line = original_lines[i]
            i += 1
            if line:
                paragraph_lines.append(line)
            if not line or i == len(original_lines):
                if paragraph_lines:
                    paragraphs.append(' '.join(paragraph_lines))
                break
        if i == len(original_lines):
            break
    return '\n'.join(paragraphs)

