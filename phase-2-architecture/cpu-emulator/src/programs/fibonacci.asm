; fibonacci.asm — Compute first 10 Fibonacci numbers and store in memory
;
; Algorithm:
;   F(0) = 0, F(1) = 1, F(n) = F(n-1) + F(n-2)
;
; Register usage:
;   R0 = F(n-2)  (previous-previous)
;   R1 = F(n-1)  (previous)
;   R2 = F(n)    (current)
;   R3 = loop counter (0..9)
;   R4 = memory address to write next value (starts at 0x0100)
;   R5 = scratch / 10 (loop limit)

    MOV R0, 0         ; F(0) = 0
    MOV R1, 1         ; F(1) = 1
    MOV R3, 0         ; counter = 0
    MOV R4, 0x0100    ; base memory address = 256

    MOV R5, 10        ; loop limit

; Store F(0) and F(1) first
    STORE [R4], R0
    PRINT R0
    MOV R2, 2
    ADD R4, R2
    STORE [R4], R1
    PRINT R1
    MOV R2, 2
    ADD R4, R2

    MOV R3, 2         ; counter = 2 (we already stored 2 values)

loop:
    CMP R3, R5
    JEQ done

    MOV R2, R0        ; R2 = R0 (F(n-2))
    ADD R2, R1        ; R2 = R0 + R1 = F(n)

    STORE [R4], R2    ; store to memory
    PRINT R2          ; print the value

    MOV R0, R1        ; shift: R0 = old R1
    MOV R1, R2        ; R1 = new value

    MOV R6, 2
    ADD R4, R6        ; advance memory pointer by 2 bytes

    MOV R6, 1
    ADD R3, R6        ; counter++

    JMP loop

done:
    HALT
