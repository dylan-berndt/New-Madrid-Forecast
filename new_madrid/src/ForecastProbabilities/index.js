import {useEffect, useState, useRef} from 'react';
import React from 'react';
import './style.css';
import {formatDate} from '../Tools';

export default function ForecastProbabilities() {
    const [probs, setProbs] = useState(new Array(14).fill(0));
    const [dates, setDates] = useState(new Array(14).fill(0));

    useEffect(() => {
        fetch("/model/14")
        .then((response) => response.json())
        .then((data) => {
            setProbs(data.sequence);
            setDates(data.dates);
        })
        .catch((err) => {
            console.error(err);
        })
    }, []);

    return <div className="ForecastArea">
        {
            probs.map((element, index) => {
                return <div className="ForecastCell">
                    <p>{formatDate(dates[index], false)}</p>
                    <p>{element}%</p>
                </div>
            })
        }
    </div>
}