import torch, random

vocab_len_in = 13
beg_token_in = torch.tensor(11)
end_token_in = torch.tensor(12)

vocab_len_out = 13
beg_token_out = torch.tensor(11)
end_token_out = torch.tensor(12)

def tokenize(s,L):
    res = torch.zeros(L, dtype=int)
    s = s[:(L-2)]
    res[0] = beg_token_in
    for i in range(L-1):
        if i >= len(s):
            res[i+1] = end_token_in
            continue
        if s[i]=="+":
            res[i+1] = 10
        else:
            res[i+1] = int(s[i])
    return res

tokenize_input = tokenize
tokenize_output = tokenize

def untokenize(out):
    out = out.tolist()
    if end_token_out in out:
        out = out[:out.index(end_token_out)]
    return "".join(str(i) if i>=0 and i<=9 else "+" for i in out)


def generate_sample(tensor_size,number_magnitude):
    num1 = random.randint(0,10**(number_magnitude)-1)
    num2 = random.randint(0,10**(random.randint(1,number_magnitude))-1)
    if random.randint(1,2)==1:
        num1,num2 = num2,num1
    inp = (str(num1)+"+"+str(num2))[:tensor_size-2]

    X = tokenize_input(inp,tensor_size)
    Y = tokenize_output(str(eval(inp)),tensor_size)
    return X,Y

def generate_dataset(num_samples,tensor_size):
    X = torch.zeros((num_samples,tensor_size)).long()
    Y = torch.zeros((num_samples,tensor_size)).long()

    for i in range(num_samples):
        number_magnitude = random.randint(1,tensor_size-4)
        X[i],Y[i] = generate_sample(tensor_size, number_magnitude)
    
    return X,Y
