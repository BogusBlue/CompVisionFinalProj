function [skin_accuracy, nonskin_accuracy] = eval_module(skin_detection, threshold)

    skin_pixel_count = 0;
    total_pixel_count = 0;
    
    % Skin Patch 1: top = 119; left = 299; bottom = 177; right = 333;
    test_skin_patch1 = skin_detection(119:177, 299:333);
    skin_pixel_count = skin_pixel_count + numel(find(test_skin_patch1 > threshold));
    total_pixel_count = total_pixel_count + numel(test_skin_patch1);
    
    % Skin Patch 2: top = 224; left = 229; bottom = 261; right = 253;
    test_skin_patch2 = skin_detection(224:261, 229:253);
    skin_pixel_count = skin_pixel_count + numel(find(test_skin_patch2 > threshold));
    total_pixel_count = total_pixel_count + numel(test_skin_patch2);
    
    % Skin Patch 3: top = 224; left = 387; bottom = 278; right = 410;
    test_skin_patch3 = skin_detection(224:278, 387:410);
    skin_pixel_count = skin_pixel_count + numel(find(test_skin_patch3 > threshold));
    total_pixel_count = total_pixel_count + numel(test_skin_patch3);
    
    % Non-skin Patch 1: top = 332; left = 36; bottom = 457; right = 570;
    test_nonskin_patch1 = skin_detection(332:457, 36:570);
    nonskin_pixel_count = numel(find(test_nonskin_patch1 < threshold));
    total_nonskin_pixel_count = numel(test_nonskin_patch1);
    
    skin_accuracy = skin_pixel_count/total_pixel_count*100;
    nonskin_accuracy = nonskin_pixel_count/total_nonskin_pixel_count*100;

end

