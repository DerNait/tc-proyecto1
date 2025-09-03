class State:
    
    def __init__(self, state_id):
        self.id = state_id
        self.transitions = {} 
        self.epsilon_transitions = []  # lista de estados alcanzables por epsilon
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
        if symbol != 'ε':  # No agregar epsilon al alfabeto
            self.alphabet.add(symbol)


class Automata:
    def __init__(self):
        self.state_counter = 0
    
    def get_new_state_id(self):
        """Genera un nuevo ID único para estados"""
        self.state_counter += 1
        return self.state_counter
    
    # 🔹 Agregar concatenaciones explícitas
    def add_concatenation_symbols(self, regex):
        """Inserta el operador '.' de concatenación explícita en la regex"""
        result = ""
        prev = None
        for c in regex:
            if prev:
                # Si prev fue símbolo, ) o * + ?, y c es símbolo o (
                if ((prev.isalnum() or prev in {')', '*', '+', '?'}) and 
                    (c.isalnum() or c == '(')):
                    result += '.'
            result += c
            prev = c
        return result

    # Precedencia de operadores
    def get_precedence(self, op):
        precedences = {'.': 2, '|': 1}
        return precedences.get(op, 0)

    # Regex a Postfix - Shunting Yard
    def regex_to_postfix(self, expression):
        expression = self.add_concatenation_symbols(expression)  # 🔹 aquí
        output = []
        stack = []
        tokens = list(expression)

        postfix_quantifiers = {'*', '+', '?'}
        binary_ops = {'.', '|'}

        for token in tokens:
            if token == '(':
                stack.append(token)
            elif token == ')':
                while stack and stack[-1] != '(':
                    output.append(stack.pop())
                stack.pop()
            elif token in postfix_quantifiers:
                output.append(token)
            elif token in binary_ops:
                while stack and self.get_precedence(stack[-1]) >= self.get_precedence(token):
                    output.append(stack.pop())
                stack.append(token)
            else:
                output.append(token)

        while stack:
            output.append(stack.pop())

        return ''.join(output)
    
    # Construcción Thompson
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
        star_afn = self.kleene_star_afn(afn)
        return self.concatenate_afn(afn, star_afn)
    
    def question_afn(self, afn):
        epsilon_afn = self.create_epsilon_afn()
        return self.union_afn(afn, epsilon_afn)
    
    def regex_to_afn(self, expression):
        postfix = self.regex_to_postfix(expression)
        print(f"Expresión postfix: {postfix}")
        
        stack = []
        for symbol in postfix:
            if symbol == '.':
                afn2 = stack.pop()
                afn1 = stack.pop()
                stack.append(self.concatenate_afn(afn1, afn2))
            elif symbol == '|':
                afn2 = stack.pop()
                afn1 = stack.pop()
                stack.append(self.union_afn(afn1, afn2))
            elif symbol == '*':
                afn = stack.pop()
                stack.append(self.kleene_star_afn(afn))
            elif symbol == '+':
                afn = stack.pop()
                stack.append(self.plus_afn(afn))
            elif symbol == '?':
                afn = stack.pop()
                stack.append(self.question_afn(afn))
            else:
                if symbol == 'ε':
                    afn = self.create_epsilon_afn()
                else:
                    afn = self.create_basic_afn(symbol)
                stack.append(afn)
        
        if len(stack) != 1:
            raise ValueError("Error en la construcción del AFN")
        return stack[0]

    # Imprimir el AFN
    def print_afn(self, afn):
        print(f"\n--- AFN ---")
        print(f"Estado inicial: {afn.start_state.id}")
        print(f"Estado final: {afn.final_state.id}")
        print("Estados y transiciones:")
        
        for state in afn.states:
            print(f"Estado {state.id}:", end="")
            if state.is_final:
                print(" (FINAL)", end="")
            print()
            for symbol, targets in state.transitions.items():
                for target in targets:
                    print(f"  --{symbol}--> {target.id}")
            for target in state.epsilon_transitions:
                print(f"  --ε--> {target.id}")

def main():
    automata = Automata()
    regex_expression = input("Ingresa una regex: ")
    postfix_expression = automata.regex_to_postfix(regex_expression)
    print("Regex en postfix:", postfix_expression)
    afn = automata.regex_to_afn(regex_expression)
    automata.print_afn(afn)

main()
