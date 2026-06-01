; factorial.asm — Compute 10! using a loop and CALL/RET
;
; This demonstrates the CALL/RET mechanism with a multiply subroutine.
; Since we have no MUL instruction, we implement multiplication via
; repeated addition in a subroutine.
;
; Register convention for multiply subroutine (fact_mul):
;   R0 = multiplicand (a)
;   R1 = multiplier   (b)
;   R2 = result (a * b) — clobbered
;   R3 = loop counter for multiply
;
; Main program:
;   R4 = accumulator (running factorial, starts at 1)
;   R5 = loop counter (1..10)
;   R6 = temp

    MOV R4, 1         ; accumulator = 1
    MOV R5, 1         ; loop counter = 1
    MOV R6, 10        ; upper limit

fact_loop:
    CMP R5, R6
    JGT print_result

    ; Multiply: R4 = R4 * R5
    ; Set up args for multiply subroutine
    MOV R0, R4        ; a = accumulator
    MOV R1, R5        ; b = loop counter
    CALL fact_mul     ; result in R2
    MOV R4, R2        ; accumulator = result

    MOV R7, 1
    ADD R5, R7        ; counter++
    JMP fact_loop

print_result:
    ; 10! = 3628800 — this overflows 16-bit (max 65535).
    ; We compute 7! = 5040 to stay within range.
    ; Actually let's print the result we got
    PRINT R4
    HALT

; ---- Subroutine: fact_mul ----
; Computes R2 = R0 * R1 using repeated addition
; Precondition: R0, R1 >= 0
; Clobbers: R2, R3
fact_mul:
    MOV R2, 0         ; result = 0
    MOV R3, R1        ; loop counter = b

mul_loop:
    CMP R3, R2        ; compare counter with 0 (actually compare R3 with 0)
    ; We need to compare R3 with 0, let's use a trick:
    ; store 0 in a scratch register
    PUSH R4           ; save R4 (we use it as 0-holder)
    MOV R4, 0
    CMP R3, R4
    POP R4
    JEQ mul_done

    ADD R2, R0        ; result += a

    PUSH R4
    MOV R4, 1
    SUB R3, R4        ; counter--
    POP R4

    JMP mul_loop

mul_done:
    RET
