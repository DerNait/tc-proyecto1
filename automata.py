import graphviz
HAS_GRAPHVIZ = True
from collections import deque
import json
import os
from copy import deepcopy

class State:
    def __init__(self, state_id):
        self.id = state_id
        self.transitions = {} 
        self.epsilon_transitions = []
        self.is_final = False

class AFN:
    def __init__(self, start_state, final_state):
        self.start_state = start_state
        self.final_state = final_state
        self.states = set()
        self.alphabet = set()
        
    def add_state(self, state):
        self.states.add(state)
        
    def add_symbol(self, symbol):
        if symbol != 'ε':
            self.alphabet.add(symbol)

class AFD:
    def __init__(self):
        self.states = set()
        self.alphabet = set()
        self.transitions = {}
        self.start_state = None
        self.final_states = set()
        
    def add_transition(self, from_state, symbol, to_state):
        if from_state not in self.transitions:
            self.transitions[from_state] = {}
        self.transitions[from_state][symbol] = to_state

    def simulate(self, input_str):
        """Simula la cadena sobre el AFD. Devuelve True si es aceptada, False en caso contrario."""
        # Empty automaton or no start state => reject
        if self.start_state is None:
            return False

        current = self.start_state
        # Empty string: accept if start state is final
        if input_str == "":
            return current in self.final_states

        for ch in input_str:
            if current not in self.transitions:
                return False
            if ch not in self.transitions[current]:
                return False
            current = self.transitions[current][ch]

        return current in self.final_states

    def minimize(self):
        """Minimiza el AFD por particiones y devuelve (min_afd, classes)
        where classes is a list of frozensets of original states representing each partition.
        """
        # Initial partition: finals and non-finals
        finals = set(self.final_states)
        non_finals = set(self.states) - finals

        P = []
        if finals:
            P.append(set(finals))
        if non_finals:
            P.append(set(non_finals))

        if not P:
            return self, [frozenset()]

        alphabet = sorted(self.alphabet)

        # Helper to get group index
        def get_group_index(state, groups):
            for i, g in enumerate(groups):
                if state in g:
                    return i
            return None

        changed = True
        while changed:
            changed = False
            newP = []
            for group in P:
                # split group by transition signatures
                sig_map = {}
                for s in group:
                    sig = []
                    for a in alphabet:
                        tgt = None
                        if s in self.transitions and a in self.transitions[s]:
                            tgt = self.transitions[s][a]
                        idx = get_group_index(tgt, P) if tgt is not None else None
                        sig.append(idx)
                    sig = tuple(sig)
                    sig_map.setdefault(sig, set()).add(s)

                if len(sig_map) == 1:
                    newP.append(group)
                else:
                    changed = True
                    for subset in sig_map.values():
                        newP.append(subset)
            P = newP

        # Build minimized AFD
        mapping = {s: i for i, g in enumerate(P) for s in g}
        min_afd = AFD()
        for i in range(len(P)):
            min_afd.states.add(i)
        min_afd.alphabet = set(self.alphabet)
        min_afd.start_state = mapping.get(self.start_state)
        for i, g in enumerate(P):
            if any(s in finals for s in g):
                min_afd.final_states.add(i)

        # Transitions: pick representative from each partition
        for i, g in enumerate(P):
            rep = next(iter(g))
            for a in alphabet:
                if rep in self.transitions and a in self.transitions[rep]:
                    tgt = self.transitions[rep][a]
                    tgt_group = mapping.get(tgt)
                    if tgt_group is not None:
                        min_afd.add_transition(i, a, tgt_group)

        classes = [frozenset(g) for g in P]
        return min_afd, classes

    def trace_derivation(self, input_str, classes=None):
        """Genera una derivacion paso a paso usando el AFD actual. Si se pasa 'classes',
        asume que los estados son clases de minimizacion y las imprime con contenido original.
        Retorna (accepted, steps) donde steps es lista de tuples (from, symbol, to).
        """
        steps = []
        if self.start_state is None:
            return False, steps

        current = self.start_state
        # initial marker
        steps.append((current, None, None))

        for ch in input_str:
            if current not in self.transitions or ch not in self.transitions[current]:
                steps.append((current, ch, None))
                return False, steps
            nxt = self.transitions[current][ch]
            steps.append((current, ch, nxt))
            current = nxt

        return (current in self.final_states), steps

class Automata:
    def __init__(self):
        self.state_counter = 0
    
    def get_new_state_id(self):
        self.state_counter += 1
        return self.state_counter

    def add_concatenation_symbols(self, regex):
        result = ""
        prev = None
        for c in regex:
            if prev:
                if ((prev.isalnum() or prev in {')', '*', '+', '?', 'ε'}) and 
                    (c.isalnum() or c == '(' or c == 'ε')):
                    result += '.'
            result += c
            prev = c
        return result

    def get_precedence(self, op):
        precedences = {'.': 2, '|': 1}
        return precedences.get(op, 0)

    def regex_to_postfix(self, expression):
        expression = self.add_concatenation_symbols(expression)
        output = []
        stack = []
        
        postfix_quantifiers = {'*', '+', '?'}
        binary_ops = {'.', '|'}

        for token in expression:
            if token == '(':
                stack.append(token)
            elif token == ')':
                while stack and stack[-1] != '(':
                    output.append(stack.pop())
                if stack:
                    stack.pop()
            elif token in postfix_quantifiers:
                output.append(token)
            elif token in binary_ops:
                while stack and stack[-1] != '(' and self.get_precedence(stack[-1]) >= self.get_precedence(token):
                    output.append(stack.pop())
                stack.append(token)
            else:
                output.append(token)

        while stack:
            output.append(stack.pop())

        return ''.join(output)
    
    def create_basic_afn(self, symbol):
        start = State(self.get_new_state_id())
        final = State(self.get_new_state_id())
        final.is_final = True
        
        if symbol not in start.transitions:
            start.transitions[symbol] = []
        start.transitions[symbol].append(final)
        
        afn = AFN(start, final)
        afn.add_state(start)
        afn.add_state(final)
        if symbol != 'ε':
            afn.add_symbol(symbol)
        
        return afn
    
    def create_epsilon_afn(self):
        start = State(self.get_new_state_id())
        final = State(self.get_new_state_id())
        final.is_final = True
        start.epsilon_transitions.append(final)
        
        afn = AFN(start, final)
        afn.add_state(start)
        afn.add_state(final)
        
        return afn
    
    def concatenate_afn(self, afn1, afn2):
        afn1.final_state.is_final = False
        afn1.final_state.epsilon_transitions.append(afn2.start_state)
        
        result = AFN(afn1.start_state, afn2.final_state)
        
        for state in afn1.states:
            result.add_state(state)
        for state in afn2.states:
            result.add_state(state)
        for symbol in afn1.alphabet:
            result.add_symbol(symbol)
        for symbol in afn2.alphabet:
            result.add_symbol(symbol)
        
        return result
    
    def union_afn(self, afn1, afn2):
        new_start = State(self.get_new_state_id())
        new_final = State(self.get_new_state_id())
        new_final.is_final = True
        
        afn1.final_state.is_final = False
        afn2.final_state.is_final = False
        
        new_start.epsilon_transitions.append(afn1.start_state)
        new_start.epsilon_transitions.append(afn2.start_state)
        
        afn1.final_state.epsilon_transitions.append(new_final)
        afn2.final_state.epsilon_transitions.append(new_final)
        
        result = AFN(new_start, new_final)
        result.add_state(new_start)
        result.add_state(new_final)
        
        for state in afn1.states:
            result.add_state(state)
        for state in afn2.states:
            result.add_state(state)
        for symbol in afn1.alphabet:
            result.add_symbol(symbol)
        for symbol in afn2.alphabet:
            result.add_symbol(symbol)
        
        return result
    
    def kleene_star_afn(self, afn):
        new_start = State(self.get_new_state_id())
        new_final = State(self.get_new_state_id())
        new_final.is_final = True
        
        afn.final_state.is_final = False
        
        new_start.epsilon_transitions.append(afn.start_state)
        new_start.epsilon_transitions.append(new_final)
        afn.final_state.epsilon_transitions.append(afn.start_state)
        afn.final_state.epsilon_transitions.append(new_final)
        
        result = AFN(new_start, new_final)
        result.add_state(new_start)
        result.add_state(new_final)
        
        for state in afn.states:
            result.add_state(state)
        for symbol in afn.alphabet:
            result.add_symbol(symbol)
        
        return result
    
    def plus_afn(self, afn):
        new_start = State(self.get_new_state_id())
        new_final = State(self.get_new_state_id())
        new_final.is_final = True
        
        afn.final_state.is_final = False
        
        new_start.epsilon_transitions.append(afn.start_state)
        afn.final_state.epsilon_transitions.append(afn.start_state)
        afn.final_state.epsilon_transitions.append(new_final)
        
        result = AFN(new_start, new_final)
        result.add_state(new_start)
        result.add_state(new_final)
        
        for state in afn.states:
            result.add_state(state)
        for symbol in afn.alphabet:
            result.add_symbol(symbol)
        
        return result
    
    def question_afn(self, afn):
        new_start = State(self.get_new_state_id())
        new_final = State(self.get_new_state_id())
        new_final.is_final = True
        
        afn.final_state.is_final = False
        
        new_start.epsilon_transitions.append(afn.start_state)
        new_start.epsilon_transitions.append(new_final)
        afn.final_state.epsilon_transitions.append(new_final)
        
        result = AFN(new_start, new_final)
        result.add_state(new_start)
        result.add_state(new_final)
        
        for state in afn.states:
            result.add_state(state)
        for symbol in afn.alphabet:
            result.add_symbol(symbol)
        
        return result
    
    def regex_to_afn(self, expression):
        postfix = self.regex_to_postfix(expression)
        
        stack = []
        for symbol in postfix:
            if symbol == '.':
                if len(stack) >= 2:
                    afn2 = stack.pop()
                    afn1 = stack.pop()
                    stack.append(self.concatenate_afn(afn1, afn2))
            elif symbol == '|':
                if len(stack) >= 2:
                    afn2 = stack.pop()
                    afn1 = stack.pop()
                    stack.append(self.union_afn(afn1, afn2))
            elif symbol == '*':
                if stack:
                    afn = stack.pop()
                    stack.append(self.kleene_star_afn(afn))
            elif symbol == '+':
                if stack:
                    afn = stack.pop()
                    stack.append(self.plus_afn(afn))
            elif symbol == '?':
                if stack:
                    afn = stack.pop()
                    stack.append(self.question_afn(afn))
            else:
                if symbol == 'ε' or symbol == 'E':
                    afn = self.create_epsilon_afn()
                else:
                    afn = self.create_basic_afn(symbol)
                stack.append(afn)
        
        if len(stack) != 1:
            raise ValueError("Error en la construccion del AFN")
        return stack[0]
    
    def epsilon_closure(self, states):
        closure = set(states)
        stack = list(states)
        
        while stack:
            state = stack.pop()
            for epsilon_state in state.epsilon_transitions:
                if epsilon_state not in closure:
                    closure.add(epsilon_state)
                    stack.append(epsilon_state)
        
        return frozenset(closure)
    
    def move(self, states, symbol):
        result = set()
        for state in states:
            if symbol in state.transitions:
                for target in state.transitions[symbol]:
                    result.add(target)
        return result
    
    def afn_to_afd(self, afn):
        print("\n=== Conversion AFN a AFD (CON epsilon-closure) ===")
        afd = AFD()
        afd.alphabet = afn.alphabet
        
        state_mapping = {}
        afd_state_counter = 0
        
        initial_closure = self.epsilon_closure({afn.start_state})
        state_mapping[initial_closure] = afd_state_counter
        afd.start_state = afd_state_counter
        afd.states.add(afd_state_counter)
        
        for state in initial_closure:
            if state.is_final:
                afd.final_states.add(afd_state_counter)
                break
        
        unprocessed = deque([initial_closure])
        processed = set()
        
        print(f"Estado inicial AFD: q{afd_state_counter} = {{{', '.join(str(s.id) for s in initial_closure)}}}")
        
        while unprocessed:
            current_states = unprocessed.popleft()
            if current_states in processed:
                continue
            processed.add(current_states)
            
            current_afd_state = state_mapping[current_states]
            
            for symbol in sorted(afd.alphabet):
                next_states = self.move(current_states, symbol)
                next_closure = self.epsilon_closure(next_states)
                
                if next_closure:
                    if next_closure not in state_mapping:
                        afd_state_counter += 1
                        state_mapping[next_closure] = afd_state_counter
                        afd.states.add(afd_state_counter)
                        unprocessed.append(next_closure)
                        
                        for state in next_closure:
                            if state.is_final:
                                afd.final_states.add(afd_state_counter)
                                break
                        
                        print(f"Nuevo estado AFD: q{afd_state_counter} = {{{', '.join(str(s.id) for s in next_closure)}}}")
                    
                    afd.add_transition(current_afd_state, symbol, state_mapping[next_closure])
                    print(f"  Transicion: q{current_afd_state} --{symbol}--> q{state_mapping[next_closure]}")
        
        return afd
    
    def save_automaton_to_json(self, automaton, filename, automaton_type="AFD"):
        """
        Guarda AFN/AFD como un JSON cuya *carga útil* son strings en el
        formato visual pedido (headers por línea y elementos en una sola línea):
        - ESTADOS = {0, 1, ..., n}
        - SIMBOLOS = {a, b, c, ..., z}
        - INICIO = {0}
        - ACEPTACION = {0, 1, ..., n}
        - TRANSICIONES = {(0, a, 1), (0, b, 2), (3, b, n), ...}

        Nota: Para AFN, se incluyen transiciones ε como (i, ε, j).
        """
        def braced(items):
            return "{%s}" % ", ".join(items)

        if automaton_type == "AFN":
            estados = sorted(s.id for s in automaton.states)
            simbolos = sorted(list(automaton.alphabet))
            inicio = automaton.start_state.id
            aceptacion = sorted(s.id for s in automaton.states if s.is_final)

            trans_list = []
            for s in automaton.states:
                for sym, targets in s.transitions.items():
                    for t in targets:
                        trans_list.append((s.id, sym, t.id))
                for t in s.epsilon_transitions:
                    trans_list.append((s.id, "ε", t.id))

            trans_list.sort(key=lambda x: (x[0], str(x[1]), x[2]))

        else:
            estados = sorted(list(automaton.states))
            simbolos = sorted(list(automaton.alphabet))
            inicio = automaton.start_state
            aceptacion = sorted(list(automaton.final_states))

            trans_list = []
            for from_state in sorted(automaton.transitions.keys()):
                for sym in sorted(automaton.transitions[from_state].keys()):
                    to_state = automaton.transitions[from_state][sym]
                    trans_list.append((from_state, sym, to_state))

        estados_str = braced([str(s) for s in estados])
        simbolos_str = braced(simbolos)
        inicio_str = braced([str(inicio)])
        aceptacion_str = braced([str(s) for s in aceptacion])
        transiciones_str = "{%s}" % ", ".join(f"({i}, {a}, {j})" for i, a, j in trans_list)

        payload = {
            "ESTADOS": estados_str,
            "SIMBOLOS": simbolos_str,
            "INICIO": inicio_str,
            "ACEPTACION": aceptacion_str,
            "TRANSICIONES": transiciones_str,
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    
    def visualize_afn(self, afn, filename="afn_graph"):
        if not HAS_GRAPHVIZ:
            print("graphviz no disponible: se omite la visualizacion del AFN")
            return
        dot = graphviz.Digraph(comment='AFN', format='png')
        dot.attr(rankdir='LR', size='10,8')
        dot.attr('node', shape='circle', style='filled', fillcolor='lightblue')
        
        for state in afn.states:
            if state.is_final:
                dot.node(str(state.id), str(state.id), shape='doublecircle', fillcolor='lightgreen')
            else:
                dot.node(str(state.id), str(state.id))
        
        dot.node('start', '', shape='plaintext')
        dot.edge('start', str(afn.start_state.id))
        
        for state in afn.states:
            for symbol, targets in state.transitions.items():
                for target in targets:
                    dot.edge(str(state.id), str(target.id), label=symbol)
            for target in state.epsilon_transitions:
                dot.edge(str(state.id), str(target.id), label='ε', style='dashed')
        
        try:
            result = dot.render(filename, format='png', cleanup=True)
            print(f"AFN visualizado en: {result}")
        except Exception as e:
            print(f"Error al generar AFN: {e}")
    
    def visualize_afd(self, afd, filename="afd_graph"):
        if not HAS_GRAPHVIZ:
            print("graphviz no disponible: se omite la visualizacion del AFD")
            return
        dot = graphviz.Digraph(comment='AFD', format='png')
        dot.attr(rankdir='LR', size='10,8')
        dot.attr('node', shape='circle', style='filled', fillcolor='lightblue')
        
        for state in sorted(afd.states):
            if state in afd.final_states:
                dot.node(f'q{state}', f'q{state}', shape='doublecircle', fillcolor='lightgreen')
            else:
                dot.node(f'q{state}', f'q{state}')
        
        dot.node('start', '', shape='plaintext')
        dot.edge('start', f'q{afd.start_state}')
        
        for from_state in sorted(afd.transitions.keys()):
            for symbol in sorted(afd.transitions[from_state].keys()):
                to_state = afd.transitions[from_state][symbol]
                dot.edge(f'q{from_state}', f'q{to_state}', label=symbol)
        
        try:
            result = dot.render(filename, format='png', cleanup=True)
            print(f"AFD visualizado en: {result}")
        except Exception as e:
            print(f"Error al generar AFD: {e}")

    def minimize_afd(self, afd: AFD, keep_trap: bool = False) -> AFD:
        """Minimiza un AFD usando Hopcroft. Si keep_trap=False, intenta remover trampa al final."""
        afd_tot = deepcopy(afd)
        Q = set(afd_tot.states)
        F = set(afd_tot.final_states)
        A = set(afd_tot.alphabet)
        delta = afd_tot.transitions

        if not Q:
            min_dfa = AFD()
            min_dfa.alphabet = set(A)
            return min_dfa

        nonF = Q - F
        P = []
        if F: P.append(F)
        if nonF: P.append(nonF)

        W = []
        if F and nonF: W.append(F if len(F) <= len(nonF) else nonF)
        elif F: W.append(F)
        elif nonF: W.append(nonF)

        reverse_delta = {c: {} for c in A}
        for q in Q:
            for c in A:
                t = delta.get(q, {}).get(c, None)
                if t is None:
                    continue
                reverse_delta[c].setdefault(t, set()).add(q)

        while W:
            A_block = W.pop()
            for c in A:
                X = set()
                for t in A_block:
                    if t in reverse_delta[c]:
                        X |= reverse_delta[c][t]
                if not X:
                    continue
                new_P = []
                for Y in P:
                    inter, diff = Y & X, Y - X
                    if inter and diff:
                        new_P += [inter, diff]
                        if Y in W:
                            W.remove(Y); W += [inter, diff]
                        else:
                            W.append(inter if len(inter) <= len(diff) else diff)
                    else:
                        new_P.append(Y)
                P = new_P

        block_id = {}
        for i, block in enumerate(P):
            for s in block:
                block_id[s] = i

        min_dfa = AFD()
        min_dfa.alphabet = set(A)
        min_dfa.states = set(range(len(P)))
        min_dfa.start_state = block_id[afd_tot.start_state]
        min_dfa.final_states = {block_id[s] for s in F}

        for i, block in enumerate(P):
            rep = next(iter(block))
            for c in A:
                to_rep = delta.get(rep, {}).get(c, None)
                if to_rep is None:
                    continue
                j = block_id[to_rep]
                min_dfa.add_transition(i, c, j)

        self._remove_unreachable_states(min_dfa)

        if not keep_trap:
            self._remove_dead_trap(min_dfa)

        classes = [frozenset(block) for block in P]
        return min_dfa, classes
    
    def _remove_unreachable_states(self, afd: AFD):
        if afd.start_state is None:
            return
        reachable = set()
        stack = [afd.start_state]
        while stack:
            s = stack.pop()
            if s in reachable: 
                continue
            reachable.add(s)
            for a in afd.alphabet:
                t = afd.transitions.get(s, {}).get(a, None)
                if t is not None and t not in reachable:
                    stack.append(t)
        # filtrar
        afd.states = afd.states & reachable
        afd.final_states = afd.final_states & reachable
        afd.transitions = {s: {a: t for a, t in trans.items() if t in afd.states}
                        for s, trans in afd.transitions.items() if s in afd.states}

    def _detect_trap_state(self, afd: AFD):
            for s in afd.states:
                if s in afd.final_states:
                    continue
                # trampa = no-aceptación y todas las salidas regresan a sí mismo
                is_trap = True
                for a in afd.alphabet:
                    t = afd.transitions.get(s, {}).get(a, None)
                    if t != s:
                        is_trap = False
                        break
                if is_trap:
                    return s
            return None

    def _remove_dead_trap(self, afd: AFD):
            trap = self._detect_trap_state(afd)
            if trap is None or trap == afd.start_state:
                return
            # quitar referencias al trap y el estado
            for s in list(afd.transitions.keys()):
                for a in list(afd.transitions[s].keys()):
                    if afd.transitions[s][a] == trap:
                        del afd.transitions[s][a]
                if not afd.transitions[s]:
                    del afd.transitions[s]
            if trap in afd.states:
                afd.states.remove(trap)
            if trap in afd.final_states:
                afd.final_states.remove(trap)

def main():
    automata = Automata()
    
    print("=== Conversor Regex a Automatas (CON epsilon-closure) ===")
    regex = input("\nIngrese la expresion regular: ")
    
    try:
        print("\n" + "="*50)
        print("PASO 1: Regex a Postfix (Shunting Yard)")
        print("="*50)
        postfix = automata.regex_to_postfix(regex)
        print("Expresión resultante (Postfix):", postfix)

        print("\n" + "="*50)
        print("PASO 2: Construccion del AFN (Thompson)")
        print("="*50)
        afn = automata.regex_to_afn(regex)
        automata.save_automaton_to_json(afn, "afn.json", "AFN")
        print("AFN guardado en 'afn.json'")
        
        print("\n" + "="*50)
        print("PASO 3: Conversion AFN a AFD (Subconjuntos)")
        print("="*50)
        afd = automata.afn_to_afd(afn)
        automata.save_automaton_to_json(afd, "afd.json", "AFD")
        print("AFD guardado en 'afd.json'")
        
        print("\n" + "="*50)
        print("PASO 4: Visualizacion")
        print("="*50)
        automata.visualize_afn(afn, "afn_graph")
        automata.visualize_afd(afd, "afd_graph")
        
        print("\n" + "="*50)
        print("PASO 5: Minimizacion del AFD (Hopcroft)")
        print("="*50)
        afd_min, afd_min_classes = automata.minimize_afd(afd)
        automata.save_automaton_to_json(afd_min, "afd_min.json", "AFD")
        print("AFD minimizado guardado en 'afd_min.json'")

        print("\n" + "="*50)
        print("PASO 6: Visualizacion del AFD minimizado")
        print("="*50)
        automata.visualize_afd(afd_min, "afd_min_graph")

        print("\nArchivos generados:")
        print("- afn.json (descripcion del AFN)")
        print("- afd.json (descripcion del AFD)")
        print("- afd_min.json (descripcion del AFD minimizado)")
        print("- afn_graph.png (visualizacion del AFN)")
        print("- afd_graph.png (visualizacion del AFD)")
        print("- afd_min_graph.png (visualizacion del AFD minimizado)")

        # Interactively read strings to simulate and show derivations step-by-step
        print("\nIngrese las cadenas a simular (una por línea). Deje línea vacía para terminar.")
        simulations = []
        while True:
            try:
                s = input("cadena> ").strip()
            except EOFError:
                break
            if s == "":
                break
            simulations.append(s)

        results = []
        # Use the Hopcroft-minimized AFD and its classes for all simulations
        for s in simulations:
            accepted, steps = afd_min.trace_derivation(s, classes=afd_min_classes)
            results.append((s, accepted, steps))

            # Print derivation step-by-step to console
            print(f"\nCadena: '{s}' -> {'ACEPTADA' if accepted else 'RECHAZADA'}")
            print("Clases de estados (minimizado):")
            for i, c in enumerate(afd_min_classes):
                print(f"  C{i} = {{{', '.join(str(st) for st in sorted(c))}}}")
            print("Derivacion:")
            for step in steps:
                frm, sym, to = step
                if sym is None:
                    print(f"  Estado inicial: C{frm} (contiene: {{{', '.join(str(st) for st in sorted(afd_min_classes[frm]))}}})")
                elif to is None:
                    print(f"  Desde C{frm} --{sym}--> (no definido) => RECHAZADA")
                else:
                    print(f"  C{frm} --{sym}--> C{to} (contiene: {{{', '.join(str(st) for st in sorted(afd_min_classes[to]))}}})")

        # Summary on console
        print("\nResumen de simulaciones:")
        for s, acc, _ in results:
            print(f"  '{s}': {'ACEPTADA' if acc else 'RECHAZADA'}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()