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
  def parse_within(bash_str):
    parsed = { 'type': 'UNKNOWN', 'children': [] } # Start with nothing
    phase = 0
    step1 = None
    step2 = None
    step3 = None
    try:
      # Try and do real parse
      step1 = subprocess.check_output(
        './app',
        stderr=subprocess.DEVNULL,
        input=bash_str.encode('utf-8')
      )
      phase = 1
      # print(json.dumps(json.loads(step1.decode('utf-8'))), flush=True)
      step2 = subprocess.check_output(
        ['jq', '-c', '--from-file', './filter-1.jq'],
        stderr=subprocess.DEVNULL,
        input=step1
      )
      phase = 2
      # print(json.dumps(json.loads(step2.decode('utf-8'))), flush=True)
      step3 = subprocess.check_output(
        ['jq', '-c', '--from-file', './filter-2.jq'],
        stderr=subprocess.DEVNULL,
        input=step2
      )
      phase = 3
      parsed = json.loads(step3.decode('utf-8'))
    except Exception as ex:
      return { 'type': 'UNKNOWN', 'children': [] }
    return parsed


  def parse_embedded_bash(node):
    if node is None or 'type' not in node:
      return node
    
    if node['type'] == 'MAYBE-BASH':
      # Send this over to our haskell parser
      return parse_within(node['value'])
    
    new_children = []
    for child in node['children']:
      new_children.append(parse_embedded_bash(child))
    
    node['children'] = new_children
    return node


  def process(line):
    return json.dumps(
      parse_embedded_bash(json.loads(line.strip()))
    )


  pool = multiprocessing.Pool()

  with lzma.open('/mnt/hgfs/SharedDirectory/datasets/2-phase-2-dockerfile-asts/dataset.jsonl.xz', mode='wt') as out_file:
    with lzma.open('/mnt/hgfs/SharedDirectory/datasets/1-phase-1-dockerfile-asts/dataset.jsonl.xz', mode='rt') as file:
      
      all_lines = file.readlines()

      # results = []
      # for line in all_lines:
      #   result = process(line)
      #   count_jiexi += 1
      #   print(f'jiexishu:{count_jiexi}')
      #   results.append(result)

      results = pool.imap(process, all_lines, chunksize=500)

      for result in tqdm.tqdm(results, total=len(all_lines), desc="Generating"):
        out_file.write('{}\n'.format(result))
        count_jiexi_all += 1

print(f'解析个数={count_jiexi_all}')