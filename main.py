# public
from os.path import join, dirname
import trainer as train_manager
from options import Options


options_handler = Options()
options = options_handler.parse()

if __name__ == "__main__":
    if options.phase in ['test']:
        print('load checkpoint from {}'.format(options.resume))
        options = options_handler.update_opt_from_json(join(dirname(options.resume), 'flags.json'), options)
        # options.split='test'
    trainer = train_manager.Trainer(options)
    print(trainer.opt.phase, '-->', trainer.opt.runsPath)

    if trainer.opt.phase in['train']:
        trainer.train()
    elif trainer.opt.phase in ['test']:
        trainer.eval()
