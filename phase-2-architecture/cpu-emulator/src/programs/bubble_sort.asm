; bubble_sort.asm — Sort 8 numbers using bubble sort
;
; Numbers are stored in memory starting at address 0x0200:
;   0x0200: 42, 0x0202: 7, 0x0204: 99, 0x0206: 1,
;   0x0208: 55, 0x020A: 23, 0x020C: 88, 0x020E: 14
;
; We pre-load them via MOV + STORE before sorting.
;
; Bubble sort:
;   for i in 0..7:
;       for j in 0..7-i-1:
;           if arr[j] > arr[j+1]: swap
;
; Register usage:
;   R0 = base address (0x0200)
;   R1 = outer loop counter i (0..7)
;   R2 = inner loop counter j (0..7-i-1)
;   R3 = array[j] address = R0 + 2*j
;   R4 = array[j]
;   R5 = array[j+1]
;   R6 = limit (8 - i - 1)
;   R7 = scratch

    ; ---- Initialize array in memory ----
    MOV R0, 0x0200    ; base address

    MOV R7, 42
    STORE [R0], R7
    MOV R7, 2
    ADD R0, R7

    MOV R7, 7
    STORE [R0], R7
    MOV R7, 2
    ADD R0, R7

    MOV R7, 99
    STORE [R0], R7
    MOV R7, 2
    ADD R0, R7

    MOV R7, 1
    STORE [R0], R7
    MOV R7, 2
    ADD R0, R7

    MOV R7, 55
    STORE [R0], R7
    MOV R7, 2
    ADD R0, R7

    MOV R7, 23
    STORE [R0], R7
    MOV R7, 2
    ADD R0, R7

    MOV R7, 88
    STORE [R0], R7
    MOV R7, 2
    ADD R0, R7

    MOV R7, 14
    STORE [R0], R7

    ; Reset base address
    MOV R0, 0x0200

    ; ---- Bubble sort ----
    MOV R1, 0         ; outer loop i = 0
    MOV R6, 8         ; array size

outer_loop:
    ; Check i < 7
    MOV R7, 7
    CMP R1, R7
    JEQ print_array

    ; inner limit = 7 - i  (number of comparisons in this pass)
    MOV R6, 7
    SUB R6, R1        ; R6 = 7 - i (note: this computes R6 = R6 - R1, i.e., 7 - i)

    MOV R2, 0         ; j = 0

inner_loop:
    ; Check j < 7 - i
    CMP R2, R6
    JEQ next_outer

    ; Compute address of arr[j]: R3 = R0 + 2*j
    MOV R3, R2
    SHL R3, 1         ; R3 = 2*j
    ADD R3, R0        ; R3 = base + 2*j

    ; Load arr[j] and arr[j+1]
    LOAD R4, [R3]     ; R4 = arr[j]
    MOV R7, 2
    ADD R3, R7        ; R3 = address of arr[j+1]
    LOAD R5, [R3]     ; R5 = arr[j+1]

    ; Compare: if arr[j] <= arr[j+1], no swap
    CMP R4, R5
    JLT no_swap
    JEQ no_swap

    ; Swap: arr[j] = R5, arr[j+1] = R4
    MOV R7, 2
    SUB R3, R7        ; back to address of arr[j]
    STORE [R3], R5    ; arr[j] = arr[j+1]
    MOV R7, 2
    ADD R3, R7        ; address of arr[j+1]
    STORE [R3], R4    ; arr[j+1] = arr[j]

no_swap:
    MOV R7, 1
    ADD R2, R7        ; j++
    JMP inner_loop

next_outer:
    MOV R7, 1
    ADD R1, R7        ; i++
    JMP outer_loop

    ; ---- Print sorted array ----
print_array:
    MOV R0, 0x0200
    MOV R1, 0         ; counter
    MOV R2, 8         ; size

print_loop:
    CMP R1, R2
    JEQ done

    LOAD R3, [R0]
    PRINT R3

    MOV R4, 2
    ADD R0, R4
    MOV R4, 1
    ADD R1, R4
    JMP print_loop

done:
    HALT
