#!/Library/Frameworks/Python.framework/Versions/3.8/bin/python3
import os
import sys

#Original see here http://www.techdeviancy.com/sconsandgraphviz/sconsandgraphviz.html

nodes = ['']
edges = { }
used_substitute_set = { }

match_skip = [ ]

substitute = { '/usr/include/glib' : 'glib', '/usr/lib/glib' : 'glib',
               '/usr/include/gtk' : 'gtk', '/usr/lib/gtk' : 'gtk',
               '/usr/include/pango' : 'pango', '/usr/include/atk' : 'atk',
               '/usr/include/cairo' : 'cairo', '/usr/include/python' : 'libpython' }

# root_start_symbol_sequence = "+-."
root_start_symbol_sequence = "+-kbc"

start = 0
for line in sys.stdin:

	if(line.find(root_start_symbol_sequence) > -1):
		start = 1
		continue

	if(start == 0):
		continue

	branch_pos = line.find("+-", 2)
	if(branch_pos > -1):
		n_levels = int((branch_pos - 2) / 2)
		node = line[branch_pos + 2:-1].replace('[', '')
		node = node.replace(']', '')

		if(node[-2:] == '.o'):
			passthrough = node
			node = 'SKIP'
			
		if(node[-3:] == '.os'):
			passthrough = node
			node = 'SKIP'

		for c in match_skip:
			if(node == c):
				node = 'SKIP'
				break

		for c in substitute:
			if(node.find(c) > -1):
				node = substitute[c]
				used_substitute_set[substitute[c]] = 1

		if(len(nodes) < n_levels + 1):
			nodes.append(node)
			skip = 0
		else:
			if(nodes[n_levels] != node):
				nodes[n_levels] = node
				skip = 0
			else:
				skip = 1

		if(skip == 0):
			if( (n_levels > 0) and (nodes[n_levels] != 'SKIP') ):
				head_level = n_levels - 1
				while( (nodes[head_level] == 'SKIP') and (head_level > -1) ):
					head_level = head_level - 1

				if(head_level > -1):
					keystring = "\"%s\" -> \"%s\"" % (nodes[head_level], nodes[n_levels])
					edges[keystring] = 1
				


print ("digraph S {")
for n in used_substitute_set:
	print (' "{}" [style=filled, fillcolor=purple];'.format(n)) 

for edge in edges:
	if(edges[edge] == 1):
		print (" {};".format(edge))

print ("}")