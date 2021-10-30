import scrapy
from ..items import CrglstItem
import os
from os.path import dirname

current_dir = os.path.dirname(__file__)
top_dir = dirname(dirname(dirname(current_dir)))
csv_file = os.path.join(top_dir, 'csv_files/data.csv')


class ExploitSpider(scrapy.Spider):
    page_number = 120
    name = 'crglst'
    allowed_domains = ['https://raleigh.craigslist.org/d/cars-trucks-by-owner/search/cto']
    start_urls = ['https://raleigh.craigslist.org/d/cars-trucks-by-owner/search/cto']
    handle_httpstatus_list = [403]

    def parse(self, response):
        items = CrglstItem()
        
        #car_year_name = response.css('.hdrlnk::text').extract()
        #car_price = response.css('.result-meta .result-price').css('::text').extract()
        car_total = response.css('.totalcount::text').extract()
        car_info = response.css('.result-meta .result-price , .hdrlnk').css('::text').extract()
        append_csv_file(car_info)
        #items['car_year_name'] = car_year_name
        #items['car_price'] = car_price
        #print('{} - {}'. format(car_year_name,car_price))
        #yield items
        
        
        
        cities = ['raleigh','NewYork', 'LosAngeles', 'Chicago',	'Houston', 'Phoenix', 
                'Philadelphia', 'SanAntonio', 'SanDiego',	'Dallas', 'SanJose', 
                'Austin', 'Jacksonville', 'FortWorth','Columbus', 'Charlotte', 
                'SanFrancisco', 'Indianapolis', 'Seattle', 'Denver'
                ]
        next_page = 'https://raleigh.craigslist.org/d/cars-trucks-by-owner/search/cto?s=' + str(ExploitSpider.page_number)
        if ExploitSpider.page_number < int(car_total[0]):
            ExploitSpider.page_number = ExploitSpider.page_number + 120
            yield response.follow(next_page, callback = self.parse,dont_filter=True)
            
def append_csv_file(car_info):
    line = f"{car_info}\n"
    if not os.path.exists(csv_file):
        with open(csv_file, 'w') as _f:
            _f.write(line)
        return
    with open(csv_file, 'a') as _f:
        _f.write(line)