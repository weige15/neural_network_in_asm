%use masm
section .text
global layerForward
global outputLoss
global layerBackward

extern vecMulMatLines
extern vecSigmoid
extern vecSigmoid_d
extern vecMSE
extern vecMulVecOuter
extern vecMulScalar
extern vecMulVecHadamard
extern vecDiff
extern vecConsoleWrite
extern matConsoleWrite

; Function: layerForward
; Description:
;     Executes the forward pass of a single neural network layer.
;     The operation performed is: output = sigmoid(W * input)
; Input:
;     rdi - Pointer to input vector
;     rsi - Pointer to weights matrix
;     rcx - Vector size (number of columns in weights matrix)
;     rdx - Number of rows in weights matrix
;     r8  - Pointer to the output vector (result)
; Output:
;     The output vector (r8) contains the result after applying the sigmoid function
layerForward:
    PUSH rdi                ; Save input vector pointer (rdi) on stack
    PUSH rsi                ; Save weights matrix pointer (rsi) on stack
    
    CALL vecMulMatLines      ; Perform matrix multiplication: W * input
    MOV rdi, r8              ; Move result pointer (output vector) into rdi
    MOV rsi, r8              ; Also copy the result pointer to rsi for the next operation
    XCHG rcx, rdx            ; Exchange the column size with row size for sigmoid function
    CALL vecSigmoid          ; Apply the sigmoid function element-wise to the result
    XCHG rcx, rdx            ; Restore original row and column sizes
    
    POP rsi                 ; Restore weights matrix pointer (rsi)
    POP rdi                 ; Restore input vector pointer (rdi)
    RET                      ; Return to the caller

; Function: outputLoss
; Description:
;     Computes the Mean Squared Error (MSE) between the actual output and the expected output.
; Input:
;     rdi - Pointer to the actual output vector
;     rsi - Pointer to the expected output vector
;     rcx - Size of the vectors (number of elements)
; Output:
;     eax - Resulting MSE loss value
outputLoss:
    JMP vecMSE               ; Directly jump to the MSE calculation function

; Function: layerBackward
; Description:
;     Performs backpropagation for a single layer, updating the weights matrix based on the error
;     and gradient information. It computes the gradient of the loss with respect to weights and
;     adjusts the weights using the provided learning rate (eta).
; Input:
;     rdi - Pointer to output vector (actual output from forward pass)
;     rsi - Pointer to expected output vector
;     rcx - Size of the output vector
;     eax - Learning rate (eta)
;     r9  - Pointer to input vector (input from the forward pass)
;     rdx - Input size (number of elements in input vector)
;     r10 - Pointer to the weights matrix
; Output:
;     The weights matrix is updated in place.
layerBackward:
    PUSH rdi                ; Save output pointer
    PUSH rcx                ; Save output size
    PUSH rdx                ; Save input size
    PUSH r8                 ; Save temporary register
    PUSH rax                ; Save learning rate (eta)
    PUSH rsi                ; Save expected output pointer

    PUSH rsi                ; Save expected output pointer again for restoration later
    MOV rsi, tmpOut         ; Use tmpOut as a temporary space for storing derivative
    CALL vecSigmoid_d       ; Compute sigmoid derivative of output: tmpOut = Sigmoid'(output)
    POP rsi                 ; Restore expected output pointer
    
    CALL vecDiff            ; Compute the error: output = output - expected
    
    PUSH rsi                ; Save expected output pointer again
    MOV rsi, tmpOut         ; Load the derivative from tmpOut
    CALL vecMulVecHadamard  ; Element-wise multiply error with sigmoid derivative: output = (output - expected) * Sigmoid'(output)
    POP rsi                 ; Restore expected output pointer
    
    CALL vecMulScalar       ; Scale the result by learning rate (eta): output = eta * (output - expected) * Sigmoid'(output)
    
    MOV rsi, rdi            ; Prepare output vector for the next operation
    MOV rdi, r9             ; Load the input vector (from forward pass) into rdi
    XCHG rcx, rdx           ; Exchange input size and output size for the outer product
    MOV r8, tmpW            ; Use tmpW to store the weight adjustments (outer product result)
    
    ; Perform outer product between output and input to compute weight adjustments
    CALL vecMulVecOuter     ; tmpW = output * input^T
    
    ; Calculate the number of elements in the weights matrix (rows * columns)
    MOV rax, rdx
    MUL rcx                 ; rax = rows * columns (total size of the weights matrix)
    MOV rcx, rax            ; Move the total size into rcx
    MOV rdi, r10            ; Set rdi to the original weights matrix pointer
    MOV rsi, tmpW           ; Load the weight adjustments from tmpW
    CALL vecDiff            ; Update the weights: weights -= tmpW (element-wise subtraction)
    
    POP rsi                 ; Restore expected output pointer
    POP rax                 ; Restore learning rate (eta)
    POP r8                  ; Restore temporary register
    POP rdx                 ; Restore input size
    POP rcx                 ; Restore output size
    POP rdi                 ; Restore output vector pointer
    RET                      ; Return to the caller

; Function: debugTmpOut
; Description:
;     Prints the contents of tmpOut for debugging purposes.
; Input:
;     None (implicitly uses the tmpOut buffer)
debugTmpOut:
    PUSH rdi                ; Save current rdi value
    PUSH rcx                ; Save current rcx value
    MOV rdi, tmpOut         ; Set rdi to point to tmpOut
    MOV rcx, 2              ; Set size to print 2 elements of tmpOut
    CALL vecConsoleWrite    ; Call function to print the values in tmpOut
    POP rcx                 ; Restore rcx
    POP rdi                 ; Restore rdi
    RET                      ; Return to the caller

; Function: debugTmpW
; Description:
;     Prints the contents of tmpW (weights adjustments) for debugging purposes.
; Input:
;     None (implicitly uses the tmpW buffer)
debugTmpW:
    PUSH rdi                ; Save current rdi value
    PUSH rcx                ; Save current rcx value
    PUSH rdx                ; Save current rdx value
    MOV rdi, tmpW           ; Set rdi to point to tmpW
    MOV rcx, 3              ; Set the number of rows to print (for debugging)
    MOV rdx, 2              ; Set the number of columns to print
    CALL matConsoleWrite    ; Call function to print the values in tmpW
    POP rdx                 ; Restore rdx
    POP rcx                 ; Restore rcx
    POP rdi                 ; Restore rdi
    RET                      ; Return to the caller

section .data

tmpOut: DD 100 dup(0)        ; Temporary buffer to store intermediate output (for sigmoid derivative)

tmpW:   DD 100 dup(0)        ; Temporary buffer to store weight adjustments (outer product result)
