import tokenize
import sys

def remove_comments(source_file, dest_file):
    with open(source_file, 'r', encoding='utf-8') as f:
        source = f.read()

    import io
    io_obj = io.StringIO(source)
    
    out = ""
    last_lineno = -1
    last_col = 0
    
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        
        if token_type == tokenize.COMMENT:
            pass
        else:
            out += token_string
            
        last_col = end_col
        last_lineno = end_line

    # Remove completely empty lines that were just comments
    out_lines = []
    for line in out.split('\n'):
        if line.strip() == '':
            # Only keep empty lines if they were empty in original?
            # Actually just keep it simple, let's keep empty lines for now, or strip them if they are completely empty.
            pass
        out_lines.append(line)

    # Better: just write out and then we can use a formatter
    with open(dest_file, 'w', encoding='utf-8') as f:
        f.write(out)

if __name__ == "__main__":
    remove_comments(sys.argv[1], sys.argv[2])
