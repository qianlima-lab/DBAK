function average = DBA(sequences)
    if(size(sequences , 1) == 1)
        average = sequences
        return ;
    end
    average = sequences(medoidIndex(sequences) , :);
    preaverage = average;
	for i=1:15
		average=DBA_one_iteration(average,sequences);
        if preaverage == average
            %fprintf('DBA Converged\n');
            break;
        end
        preaverage = average;
	end
end

function index = medoidIndex(sequences) 
    dist_dtw = squareform(pdist(sequences,@mydtw));
    dist_dtw = sum(dist_dtw , 2);
    [~, index] = min(dist_dtw);
end

function dist = mydtw(x , y)
    n = size(y , 1);
    dist = zeros(n ,  1);
    for i = 1 : n
        [dist(i) , ~ , ~] = dtw(x ,y(i, :),'squared');
    end
end

function average = DBA_one_iteration(averageS,sequences)

	tupleAssociation = cell (1, size(averageS,2));
	for t=1:size(averageS,2)
		tupleAssociation{t}=[];
	end

	n = size(sequences , 1);
    for j = 1: n
        y = sequences(j , :);
        [~ , ix , iy] = dtw(averageS ,y,'squared');  
        npath = size(ix);
        for i = 1 : npath
            tupleAssociation{ix(i)}(end + 1) = y(iy(i));
        end
    end    

	for t=1:size(averageS,2)
	   averageS(t) = mean(tupleAssociation{t});
	end
	   
	average = averageS;
end