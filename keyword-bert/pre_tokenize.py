# -*- coding:utf-8 -*-
"""
Wilson Tam: This module tokenize questions and writes to output
"""

import sys
#reload(sys)
#sys.setdefaultencoding('utf8')
import json

import tensorflow as tf
import codecs
import gzip
import tokenization

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None, "Plain Input file (or comma-separated list of files).")
flags.DEFINE_string("output_file", None, "Output file.")
flags.DEFINE_string("vocab_file", "pre_trained/vocab.txt", "Output file in Squad format.")
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    tokenizer = tokenization.FullTokenizer(FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Reading from input files ***")
    for input_file in input_files:
        tf.logging.info("  %s", input_file)

    with codecs.open(FLAGS.output_file, "w", encoding="utf-8") as output_json:
        for input_file in input_files:
            if input_file.endswith("gz"):
                reader = gzip.open(input_file, 'rb') 
            else:
                reader = tf.gfile.GFile(input_file, "r")
  
            jj = 0
            while True:
                line = tokenization.convert_to_unicode(reader.readline())
                if not line:
                    break
                line = line.strip()
                line=line.split('\t')
                label=line[0]
                q=line[1]
                a=line[2]
                q_=" ".join(tokenizer.tokenize(q))
                a_=" ".join(tokenizer.tokenize(a)) 
                output_json.write(label+'\t'+ q_ +'\t'+ a_+'\n')

                  
                '''
                #sample = json.loads(line)
    
                tf.logging.info("line {}".format(jj))
                if line!='':
                    question = " ".join(tokenizer.tokenize(line))
                    #output_json.write(question.encode('utf-8'))
                    output_json.write(question)
                    output_json.write("\t" + " ".join([str(x) for x in tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line))]))
                    #output_json.write(question)
                    output_json.write("\n")
                '''
                '''
                if sample['question']:
                    question = " ".join(tokenizer.tokenize(sample['question']))
                    #output_json.write(question.encode('utf-8'))
                    output_json.write(question)
                    output_json.write("\t" + " ".join([str(x) for x in tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sample['question']))]))
                    #print(tokenization.printable_text(question))
                    #output_json.write(tokenization.printable_text(question))
                    #output_json.write(question)
                    output_json.write("\n")
                '''
                jj += 1

if __name__ == '__main__':
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    tf.app.run()
