import graphviz
from collections import deque
import json
import os

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
    
    def save_automaton_to_file(self, automaton, filename, automaton_type="AFD"):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"=== {automaton_type} ===\n")
            
            if automaton_type == "AFN":
                states_ids = sorted([s.id for s in automaton.states])
                f.write(f"ESTADOS = {{{', '.join(map(str, states_ids))}}}\n")
                f.write(f"SIMBOLOS = {{{', '.join(sorted(automaton.alphabet))}}}\n")
                f.write(f"INICIO = {{{automaton.start_state.id}}}\n")
                
                final_states = [s.id for s in automaton.states if s.is_final]
                f.write(f"ACEPTACION = {{{', '.join(map(str, final_states))}}}\n")
                
                f.write("TRANSICIONES = {")
                transitions = []
                for state in automaton.states:
                    for symbol, targets in state.transitions.items():
                        for target in targets:
                            transitions.append(f"({state.id}, {symbol}, {target.id})")
                    for target in state.epsilon_transitions:
                        transitions.append(f"({state.id}, ε, {target.id})")
                f.write(', '.join(transitions))
                f.write("}\n")
            else:
                f.write(f"ESTADOS = {{{', '.join(f'q{s}' for s in sorted(automaton.states))}}}\n")
                f.write(f"SIMBOLOS = {{{', '.join(sorted(automaton.alphabet))}}}\n")
                f.write(f"INICIO = {{q{automaton.start_state}}}\n")
                f.write(f"ACEPTACION = {{{', '.join(f'q{s}' for s in sorted(automaton.final_states))}}}\n")
                
                f.write("TRANSICIONES = {")
                transitions = []
                for from_state in sorted(automaton.transitions.keys()):
                    for symbol in sorted(automaton.transitions[from_state].keys()):
                        to_state = automaton.transitions[from_state][symbol]
                        transitions.append(f"(q{from_state}, {symbol}, q{to_state})")
                f.write(', '.join(transitions))
                f.write("}\n")
    
    def visualize_afn(self, afn, filename="afn_graph"):
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
        automata.save_automaton_to_file(afn, "afn.txt", "AFN")
        print("AFN guardado en 'afn.txt'")
        
        print("\n" + "="*50)
        print("PASO 3: Conversion AFN a AFD (Subconjuntos)")
        print("="*50)
        afd = automata.afn_to_afd(afn)
        automata.save_automaton_to_file(afd, "afd.txt", "AFD")
        print("AFD guardado en 'afd.txt'")
        
        print("\n" + "="*50)
        print("PASO 4: Visualizacion")
        print("="*50)
        automata.visualize_afn(afn, "afn_graph")
        automata.visualize_afd(afd, "afd_graph")
        
        print("\nArchivos generados:")
        print("- afn.txt (descripcion del AFN)")
        print("- afd.txt (descripcion del AFD)")
        print("- afn_graph.png (visualizacion del AFN)")
        print("- afd_graph.png (visualizacion del AFD)")
        # --- Simulaciones ---
        sim_file = "simulations.txt"
        simulations = []
        if os.path.exists(sim_file):
            print(f"Leyendo simulaciones desde '{sim_file}'...")
            with open(sim_file, 'r', encoding='utf-8') as sf:
                for line in sf:
                    s = line.strip()
                    if s == '':
                        continue
                    simulations.append(s)
        else:
            print("No se encontró 'simulations.txt'. Puedes ingresar cadenas manualmente (una por línea). Termina con una línea vacía.")
            while True:
                try:
                    s = input()
                except EOFError:
                    break
                if s == "":
                    break
                simulations.append(s)

        results = []
        for s in simulations:
            accepted = afd.simulate(s)
            results.append((s, accepted))

        # Guardar resultados en afd_output.txt
        out_file = "afd_output.txt"
        with open(out_file, 'w', encoding='utf-8') as of:
            of.write(f"ESTADO INICIAL = q{afd.start_state}\n")
            of.write(f"ESTADOS ACEPTACION = {{{', '.join(f'q{st}' for st in sorted(afd.final_states))}}}\n")
            of.write("RESULTADOS = {\n")
            for s, acc in results:
                of.write(f"  ('{s}', {'ACEPTADA' if acc else 'RECHAZADA'})\n")
            of.write("}\n")

        print(f"Resultados de simulacion guardados en '{out_file}'")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()