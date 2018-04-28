#include "darknet.h"
#include "utils.h"

void netdump( char *cfgfile, char *weightfile )
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 2);
}

void run_netdump(int argc, char **argv)
{
    if(argc < 1){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[2];
    char *weights = (argc > 3) ? argv[3] : 0;
    netdump(cfg, weights);
}


