from torch.autograd import Function, Variable


class HardSigmoid(Function):
    @staticmethod
    def forward(ctx, input):
        output = (0.2 * input) + 0.5
        output[output > 1] = 1
        output[output < 0] = 0
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables
        grad_input = input.new(input.size())
        grad_input.data.fill_(0.2)
        grad_input[input <= -2.5] = 0
        grad_input[input >= 2.5] = 0
        return grad_input
