#! /usr/bin/env python3

import text_preproc.text_preproc as text_preproc
import annotation.annotation as annotation
import character_extraction.character_extraction as character_extraction
import character_extraction.name_unification_model as model
import quote_attribution.quote_attribution as quote_attribution
import network_creation.network_creation as network_creation
import output_format.out_formatter as out_formatter

import evaluation.quotes_evaluation as quotes_evaluation
import evaluation.character_evaluation as character_evaluation

import logging
import argparse
import os
import pickle

import spacy
from spacy.tokens import DocBin


def init():
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='A tool for book processing.')
    subparsers = parser.add_subparsers(dest='action')
    
    run_parser = subparsers.add_parser('run', help='Process a book')
    run_parser.add_argument('--book', default='data/example/A_Scandal_in_Bohemia.txt', help='Path to the book text or docbin')
    run_parser.add_argument('--out', default='out', help='Path to the output directory')
    run_parser.add_argument('--model', default='models/all_data.model', help='Path to the trained model')
    run_parser.add_argument('--maxprob', type=float, default=0.9, help='The max probability of edges removed in Character Detection')
    run_parser.add_argument('--removelimit', type=int, default=3, help='The minimum number of occurences of a character to be counted')
    run_parser.add_argument('-n', '--nosave', action='store_true', help='Does not save the annotated data')
    run_parser.add_argument('--goldcharacters', help='The list of golden characters')
    run_parser.add_argument('--goldxml', help='The file annotated with golden speakers')
    
    collect_parser = subparsers.add_parser('collect', help='Collect data to train a model')
    collect_parser.add_argument('--path', default='data/data_vala', help='Path to the book directory')
    collect_parser.add_argument('-n', '--nosave', action='store_true', help='Does not save the annotated data, saves only the weights')
    
    train_parser = subparsers.add_parser('train', help='Train a model to recognize character name equality')
    train_parser.add_argument('--path', default='data/data_vala', help='Path to the directory with character pair weights')
    train_parser.add_argument('--out', default='models/all_data.model', help='Path to save the model')
    
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate the accuracy')
    evaluate_parser.add_argument('type', choices=['characters', 'quotes'], help='Choose the type of evaluation')
    evaluate_parser.add_argument('--file', required=True, help='Docbin file to be evaluated')
    evaluate_parser.add_argument('--goldxml', help='The golden data for quotes evaluation')
    evaluate_parser.add_argument('--goldcharacters', help='The golden characters for character evaluation')
    evaluate_parser.add_argument('--model', default='models/all_data.model', help='Path to the trained model')
    evaluate_parser.add_argument('--maxprob', type=float, default=0.9, help='The max probability of edges removed in Character Detection')
    evaluate_parser.add_argument('--removelimit', type=int, default=3, help='The minimum number of occurences of a character to be counted')
    
    return parser, run_parser


def main():
    parser, def_parser = init()
    args, extras = parser.parse_known_args()
    if args.action == None:
        args = def_parser.parse_args()
        args.action = 'run'
    else:
        args = parser.parse_args()
    
    
    if args.action == 'run':
        # Phase 0: get and annotate the book text
        book = args.book
        if book.split('.')[-1] == 'docbin':
            docs = annotation.FalseAnnotator().annotate(book)
        else:
            paragraphs = text_preproc.getPars(book)
            docs = annotation.Annotator().annotate(paragraphs)
        
        if not args.nosave:
            doc_bin = DocBin(store_user_data=True, docs=docs)
            doc_bin.to_disk(book.split('.')[0] + '.docbin')
            logging.info("Annotated docs saved to {}".format(book.split('.')[0] + '.docbin'))
        
        # Phase 1: extract characters
        if args.goldcharacters:
            try:
                characters = character_evaluation.CharacterEvaluator.parseCharGender(args.goldcharacters)
            except Exception as e:
                print()
                print(e)
                logging.error("""
Wrong format of golden characters file!
The file must be in a csv format with the following columns:
character id, character name, characted gender""")
                return
            character_extractor = character_extraction.FalseCharacterExtractor(docs, characters)
        else:
            character_extractor = character_extraction.CharacterExtractor(docs)
        characters = character_extractor.extractCharacters(args.model, args.maxprob, args.removelimit)
        
        # Phase 2: assign speakers to quotes
        if args.goldxml:
            quote_attributor = quote_attribution.FalseQuoteAttributor(docs, characters, args.goldxml)
        else:
            quote_attributor = quote_attribution.QuoteAttributor(docs, characters)
        assigned_speakers_docs = quote_attributor.extractSpeakers()
        
        # Phase 3: create character network
        network_creator = network_creation.NetworkCreator(assigned_speakers_docs, characters)
        cooccurrenceG, conversationG, goldConversationG = network_creator.createNetworks(golden_speakers=args.goldxml)
        genderG = network_creator.createGenderNetwork()

        # Phase Final: output results in a nice format
        base_name = os.path.basename(args.book).split('.')[0]
        out_formatter.outputThreeNetworks(conversationG, cooccurrenceG, goldConversationG, args.out, base_name)
        out_formatter.outputGender(genderG, args.out, base_name)
        out_formatter.outputCharacters(characters, args.out, base_name)
        
        
    elif args.action == 'collect':
        annotator = annotation.Annotator()
        
        for root, dirs, files in os.walk(args.path):
            for file in files:
                if not 'txt' == file.split('.')[-1]:
                    continue
                if file.split('.')[0] + '.weights' in files:
                    continue
                logging.info("Collecting data from file {}".format(os.path.join(args.path, file)))
                out_file = os.path.join(root, file.split('.')[0] + '.weights')
                paragraphs = text_preproc.getPars(os.path.join(root, file))
                docs = annotator.annotate(paragraphs)
                character_extractor = character_extraction.CharacterExtractor(docs)
                character_extractor.saveWeights(out_file)
                logging.info("Weights saved to file {}".format(out_file))
                
                if not args.nosave:
                    doc_bin = DocBin(store_user_data=True, docs=docs)
                    doc_file = os.path.join(root, file.split('.')[0] + '.docbin')
                    doc_bin.to_disk(doc_file)
                    logging.info("Annotated docs saved to file {}".format(doc_file))
        return
    
    elif args.action == 'train':
        model.trainModel(args.path, args.out)
        return
        
    elif args.action == 'evaluate':
        if args.type == 'quotes':
            if not args.goldxml:
                print("goldxml argument required for evaluation of quotes!")
                return
            docs = annotation.FalseAnnotator().annotate(args.file)
            
            if args.goldcharacters:
                characters = character_evaluation.CharacterEvaluator.parseCharGender(args.goldcharacters)
                character_extractor = character_extraction.FalseCharacterExtractor(docs, characters)
            else:
                character_extractor = character_extraction.CharacterExtractor(docs)
            characters = character_extractor.extractCharacters(args.model, args.maxprob, args.removelimit)
            
            quote_attributor = quote_attribution.QuoteAttributor(docs, characters)
            assigned_speakers_docs = quote_attributor.extractSpeakers()
            
            evaluator = quotes_evaluation.QuotesEvaluatorQuoteLi3(docs, args.goldxml, characters)
            accuracy = evaluator.evaluate()
            print("Quote Attribution accuracy evaluated.")
            print("Accuracy: {:.2f}".format(100*accuracy))
            return
        
        elif args.type == 'characters':
            if not args.goldcharacters:
                print("goldcharacters argument required for evaluation of characters!")
                return
            docs = annotation.FalseAnnotator().annotate(args.file)
            character_extractor = character_extraction.CharacterExtractor(docs)
            characters = character_extractor.extractCharacters(args.model, args.maxprob, args.removelimit)
            
            pred_dict = {}
            for i in characters:
                variants = []
                for (name, count) in characters[i][0]:
                    variants.append(name)
                pred_dict[i] = variants
    
            evaluator = character_evaluation.CharacterEvaluatorVala()
            
            try:
                gold_dict = evaluator.parseValaGold(args.goldcharacters)
            except Exception as e:
                print()
                print(e)
                logging.error("""
Wrong format of golden characters file!
The file must be in a csv format with the following columns:
character id, character name""")
                return
            
            precision = evaluator.getPrecision(pred_dict, gold_dict)
            recall = evaluator.getRecall(pred_dict, gold_dict)
            if precision and recall:
                f1 = 2/((1/precision) + (1/recall))
            else:
                f1 = 0
            print("\nCharacter Detection Evaluation -- Unweighted metric")
            print("precision: {:.2f}, recall: {:.2f}, f1: {:.2f}".format(100*precision, 100*recall, 100*f1))
            
            
            evaluator = character_evaluation.CharacterEvaluatorImportance()
            
            try:
                gold_dict = evaluator.parseCountsGold(args.goldcharacters)
            except Exception as e:
                print()
                print(e)
                logging.error("""
Wrong format of golden characters file!
The file must be in a csv format with the following columns:
character id, character name, occurence count""")
                return
            
            pred_dict = {}
            for i in characters:
                pred_dict[i] = characters[i][0]
            
            precision = evaluator.getPrecision(pred_dict, gold_dict)
            recall = evaluator.getRecall(pred_dict, gold_dict)
            if precision and recall:
                f1 = 2/((1/precision) + (1/recall))
            else:
                f1 = 0
            print("\nCharacter Detection Evaluation -- Importance-Weighted metric")
            print("precision: {:.2f}, recall: {:.2f}, f1: {:.2f}".format(100*precision, 100*recall, 100*f1))
            
    return


if __name__ == "__main__":
    main()

