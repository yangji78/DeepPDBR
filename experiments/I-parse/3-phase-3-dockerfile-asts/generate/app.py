if __name__ == '__main__':
  import os
  import sys
  import json
  import lzma
  import tqdm
  import subprocess
  import multiprocessing

  count_jiexi = 0
  count_jiexi_all = 0
  def enrich(ast_str):
    as_json = json.loads(ast_str)
    parsed = { 'type': 'UNKNOWN', 'children': [] } # Start with nothing
    try:
      # Try and do real parse
      parsed = json.loads(subprocess.check_output(
        [ 'node', '/build/app.js' ],
        input=ast_str.encode('utf-8')
      ).decode('utf-8'))
    except Exception:
      pass
    
    parsed['file_sha'] = as_json['file_sha']
    return json.dumps(parsed)


  with lzma.open('/mnt/outputs/dataset.jsonl.xz', mode='wt') as out_file:
    with lzma.open('/mnt/inputs/dataset.jsonl.xz', mode='rt') as file:
      pool = multiprocessing.Pool()
      
      all_lines = file.readlines()
      length = len(all_lines)

      results = pool.imap(enrich, all_lines, chunksize=500)

      # results = []
      # for line in all_lines:
      #   result = enrich(line)
      #   count_jiexi += 1
      #   print(f'jiexishu:{count_jiexi}')
      #   results.append(result)

      for result in tqdm.tqdm(results, total=length, desc="Generating"):
        out_file.write('{}\n'.format(result))
        count_jiexi_all += 1

print(f'解析个数={count_jiexi_all}')
