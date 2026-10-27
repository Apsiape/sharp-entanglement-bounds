"""Standalone lossless reader of the full finite parity game and raw lifts.

No source relation, negative quotient, or physical model is used to expand the
operator DAG. A matrix entry is a sign and a literal free source word. The game
itself is completely specified by the explicit variable/row arrays.
"""
import argparse
import json
from fractions import Fraction
from pathlib import Path

HERE=Path(__file__).resolve().parent


def free(word):
    stack=[]
    for letter in word:
        if stack and stack[-1]==-letter:stack.pop()
        else:stack.append(letter)
    return tuple(stack)


def expand(data):
    out=[]
    for i,node in enumerate(data['nodes']):
        tag=node[0]
        if tag=='I':
            n=node[1];matrix=(tuple(range(n)),((1,()),)*n)
        elif tag=='L':matrix=(tuple(node[1]),tuple((s,tuple(w)) for s,w in node[2]))
        else:
            assert 0<=node[1]<i and 0<=node[2]<i
            p,w=out[node[1]];q,v=out[node[2]];n=len(p);assert n==len(q)
            if tag=='D':matrix=(p+tuple(n+j for j in q),w+v)
            elif tag=='A':matrix=(tuple(n+j for j in p)+q,w+v)
            elif tag=='M':matrix=(tuple(q[j] for j in p),tuple((w[i][0]*v[j][0],free(w[i][1]+v[j][1])) for i,j in enumerate(p)))
            else:raise ValueError(tag)
        out.append(matrix)
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--variable');p.add_argument('--row',type=int)
    args=p.parse_args();data=json.loads((HERE/'TWO-COPY-FINAL-GAME.json').read_text())
    result={'format':data['format'],'counts':data['counts'],'game_rule':data['game_rule']}
    if args.variable is not None:
        names=data['variable_names']
        index=int(args.variable) if args.variable.isdigit() else names.index(args.variable)
        p,w=expand(data)[data['variable_operators'][index]]
        result['expanded_variable']={'index':index,'name':names[index],
                                     'row_to_column':p,'entries':[{'sign':s,'word':v} for s,v in w]}
    if args.row is not None:
        row=data['rows'][args.row];sign=row[0];vs=row[1:]
        result['row']={'index':args.row,'product_sign':sign,'variable_indices':vs,
                       'variable_names':[data['variable_names'][i] for i in vs],
                       'probability_of_each_incident_pair':str(Fraction(1,len(data['rows'])*len(vs)))}
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
