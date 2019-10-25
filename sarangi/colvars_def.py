import pyparsing as pp
from pyparsing import pyparsing_common as ppc

__all__ = ['load_colvars', 'handle_colvars']

pp.ParserElement.setDefaultWhitespaceChars(' \t')

vector = pp.Group(pp.Literal('(').suppress() + 
                  pp.delimitedList(ppc.number, delim=',') + 
                  pp.Literal(')').suppress())
program = pp.Forward()
block = pp.Literal('{').suppress() + program + pp.Literal('}').suppress()
name = pp.Word(pp.alphanums + '_')
# TODO: better handling of integer lists
argument = ppc.number() | block | vector | name
statement = pp.Group(name + pp.Optional(pp.White(ws=' \t').suppress() + 
                     pp.delimitedList(argument, delim=pp.White(ws=' \t'))))
program << pp.Optional(pp.LineEnd()).suppress() + \
           pp.delimitedList(statement, pp.OneOrMore(pp.LineEnd())) + \
           pp.Optional(pp.LineEnd()).suppress()


def handle_distanceVec(dv):
    #print('DV', dv)
    if dv[0][1][0] == 'atomnumbers':
        return [int(i) for i in dv[0][1][1]]
    elif dv[1][1][0] == 'atomnumbers':
        return [int(i) for i in dv[1][1][1]]
    else:
        raise ValueError('distanceVec has no group with atomnumbers')


def handle_colvar(definition):
    name = None
    value = None
    for line in definition:
        cv = {}
        if line[0] == 'name':
            if len(line) != 2:
                raise ValueError('Unexpected number of arguments in name defintion.')
            else:
                name = line[1]
        elif line[0] == 'distanceVec' or line[0] == 'distance':
            value = handle_distanceVec(line[1:])
        else:
            raise ValueError('Unregonized command: ' + str(line[0]))
    return {name: value}


def handle_colvars(data):
    'Extracts name and atomnumbers from colvar defintions.'
    res = program.parseString(data)
    colvars = {}
    for colvar in res:
        if colvar[0] == 'colvar':
            colvars.update(handle_colvar(colvar[1:]))
        else:
            print('skipping unknown command', colvar[0])
    return colvars


def load_colvars(fname):
    'Extracts name and atomnumbers from colvar file.'
    with open(fname) as f:
        cv = ''.join(f.readlines())
    return handle_colvars(cv)


if __name__ == '__main__':
    cv='''
    colvar {  
        name LEU_248_sc
        distanceVec {
            group2 { atomnumbers { 239 242 244 248 } }
            group1 { dummyAtom ( 0.0 , 0.0 , 0.0 ) }
        }
    }
    '''
    print(handle_colvars(cv))



