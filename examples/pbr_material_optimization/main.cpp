#include "../common/desktop_host.h"
#include "run.h"

int main(int argc, char* argv[]) {
    dressi_examples::DesktopHost host;
    return dressi_examples::RunPbrMaterialOptimization(
            {argv + 1, argv + argc}, host);
}
