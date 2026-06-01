#include "vm.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

static void run_file(const std::string &path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Could not open file '" << path << "'\n";
        std::exit(74);
    }
    std::ostringstream oss;
    oss << f.rdbuf();
    std::string source = oss.str();

    VM vm;
    InterpretResult result = vm.interpret(source);
    if (result == InterpretResult::COMPILE_ERROR) std::exit(65);
    if (result == InterpretResult::RUNTIME_ERROR) std::exit(70);
}

static void run_repl(void) {
    VM vm;
    std::string line;

    std::cout << "Lox-mini VM v1.0  (type 'exit' to quit)\n";
    while (true) {
        std::cout << "> ";
        if (!std::getline(std::cin, line)) {
            std::cout << "\n";
            break;
        }
        if (line == "exit" || line == "quit") break;
        if (line.empty()) continue;

        /* Each REPL line gets a fresh interpreter call.
         * Note: global variables persist across lines because we reuse the VM. */
        vm.interpret(line);
    }
}

static void usage(const char *prog) {
    std::cerr << "Usage:\n";
    std::cerr << "  " << prog << "              -- start REPL\n";
    std::cerr << "  " << prog << " <file.lox>   -- run file\n";
}

int main(int argc, char *argv[]) {
    if (argc == 1) {
        run_repl();
    } else if (argc == 2) {
        run_file(argv[1]);
    } else {
        usage(argv[0]);
        return 64;
    }
    return 0;
}
