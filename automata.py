class Automata:
    def __init__(self):
        pass
    
    # Regexp a Postfix - Shunting Yard
    def get_precedence(self, op):
        precedences = {'.': 2, '|': 1}
        return precedences.get(op, 0)

    def regex_to_postfix(self, expression):
        output = []
        stack = []
        tokens = list(expression)

        postfix_quantifiers = {'*', '+', '?'}
        binary_ops = {'.', '|'}

        for token in tokens:
            print("Token:", token)
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

            print("Output stack:", output)
            print("Operator stack:", stack)

        while stack:
            output.append(stack.pop())

        return ''.join(output)
    
    # Regexp a AFN - (Thompson o Glushkov)
    def regex_to_afn(self, expression):
        # TODO: Definir aquí regex a AFN
        return
    
    # AFN a AFD
    def afn_to_afd(self, afn):
        # TODO: Definir aquí AFN a AFD
        return

    # AFD a AFD minimal - Hopcroft
    def minimize_afd(self, afd): 
        # TODO: Definir aqui la minimización del AFD
        return

    # Simulaciones de un AFD
    def simulate_afd(self, afd):
        # TODO: Simular AFD aquí
        return

def main():
    automata = Automata()
    regex_expression = input("Ingresa una regex: ")
    postfix_expression = automata.regex_to_postfix(regex_expression)

    print("Regex en postfix:", postfix_expression)
main()
