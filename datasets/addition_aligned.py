import torch, random

vocab_len_in = 102
vocab_len_out = 12

beg_token_in = torch.tensor(100)
end_token_in = torch.tensor(101)
beg_token_out = torch.tensor(10)
end_token_out = torch.tensor(11)

def tokenize_input(s,L):
    res = torch.zeros(L, dtype=int)
    s1,s2 = s.split("+")

    s1 = (s1[::-1]+"0"*L)[:L-2]
    s2 = (s2[::-1]+"0"*L)[:L-2]
    for i, (s1,s2) in enumerate(zip(s1,s2)):
        if i == len(res):
            break
        res[i+1] = int(s1)*10+int(s2)
    res[0] = beg_token_in
    res[-1] = end_token_in
    return res
    
def tokenize_output(s,L):
    res = torch.zeros(L, dtype=int)
    s2 = s[::-1][:L-2]
    for i, si in enumerate(s2):
        if i == len(res):
            break
        res[i+1] = int(si)
    res[0] = beg_token_out
    res[-1] = end_token_out
    return res

def untokenize(out):
    out = out.tolist()[1:]
    if 11 in out:
        out = out[:out.index(11)]
    while len(out)>1 and out[-1]==0:
        out.pop()
    return "".join(str(i) for i in out[::-1])

def generate_sample(tensor_size,number_magnitude):
    num1 = random.randint(0,10**(number_magnitude))
    num2 = random.randint(0,10**(random.randint(1,number_magnitude)))
    if random.randint(1,2)==1:
        num1,num2 = num2,num1
    
    X = tokenize_input(str(num1)+"+"+str(num2),tensor_size)
    Y = tokenize_output(str(num1+num2),tensor_size)
    return X,Y

def generate_dataset(num_samples,tensor_size):
    X = torch.zeros((num_samples,tensor_size)).long()
    Y = torch.zeros((num_samples,tensor_size)).long()

    for i in range(num_samples):
        number_magnitude = random.randint(1,tensor_size)
        X[i],Y[i] = generate_sample(tensor_size, number_magnitude)
    
    return X,Y
