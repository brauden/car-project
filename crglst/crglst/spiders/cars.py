import scrapy
from ..items import CrglstItem
import os
from os.path import dirname
from urllib.parse import urljoin

current_dir = os.path.dirname(__file__)
top_dir = dirname(dirname(dirname(current_dir)))
csv_file = os.path.join(top_dir, 'csv_files/data.csv')


class ExploitSpider(scrapy.Spider):
    name = 'crglst'
    allowed_domains = ['https://www.craigslist.org/about/sites#US']
        
    start_urls = []
    handle_httpstatus_list = [403]
    
    def __init__(self):
        page_number = 120
        urls =  ['https://raleigh.craigslist.org/d/cars-trucks-by-owner/search/cto?hasPic=1']#,
#                     'https://NewYork.craigslist.org/d/cars-trucks-by-owner/search/cto?hasPic=1',
#                     'https://LosAngeles.craigslist.org/d/cars-trucks-by-owner/search/cto?hasPic=1',
#                     'https://LosAngeles.craigslist.org/d/cars-trucks-by-owner/search/cto?hasPic=1',
#                     'https://Chicago.craigslist.org/d/cars-trucks-by-owner/search/cto?hasPic=1',
#                     'https://Houston.craigslist.org/d/cars-trucks-by-owner/search/cto?hasPic=1',
#                     'https://Phoenix.craigslist.org/d/cars-trucks-by-owner/search/cto?hasPic=1',
#                     'https://Philadelphia.craigslist.org/d/cars-trucks-by-owner/search/cto?hasPic=1',
#                     'https://SanAntonio.craigslist.org/d/cars-trucks-by-owner/search/cto?hasPic=1',
#                     'https://SanDiego.craigslist.org/d/cars-trucks-by-owner/search/cto?hasPic=1',
#                     'https://Dallas.craigslist.org/d/cars-trucks-by-owner/search/cto?hasPic=1',
#                     'https://Austin.craigslist.org/d/cars-trucks-by-owner/search/cto?hasPic=1',
#                     'https://Jacksonville.craigslist.org/d/cars-trucks-by-owner/search/cto?hasPic=1',
#                     'https://Columbus.craigslist.org/d/cars-trucks-by-owner/search/cto?hasPic=1',
#                     'https://Charlotte.craigslist.org/d/cars-trucks-by-owner/search/cto?hasPic=1',
#                     'https://SanFrancisco.craigslist.org/d/cars-trucks-by-owner/search/cto?hasPic=1',
#                     'https://Indianapolis.craigslist.org/d/cars-trucks-by-owner/search/cto?hasPic=1',
#                     'https://Seattle.craigslist.org/d/cars-trucks-by-owner/search/cto?hasPic=1',
#                     'https://Denver.craigslist.org/d/cars-trucks-by-owner/search/cto?hasPic=1'
#                     ]
        for url in urls:
            self.start_urls.append(url)
            for i in range(1,25):
                next_page = url + '&s=' + str(i*page_number)
                self.start_urls.append(next_page)
            
    # def start_requests(self):
    #     for url in self.start_urls:
    #         yield Request(url, callback=self.parse)

    def parse(self, response):
        items = CrglstItem()
        car_year_name = response.css('.hdrlnk::text').extract()
        car_price = response.css('.result-meta .result-price').css('::text').extract()
        car_urls = response.xpath('//a/@data-ids').extract()

    # make each one into a full URL and add to item[]
               
        # d_car_urls = response.selector.xpath('//img/@src').extract() #('img::attr("src")').extract()
        # img_urls = [urljoin(response.url, src)
        #                 for src in d_car_urls]
        # items['image_urls'] = img_urls

        #car_total = response.css('.totalcount::text').extract()
        #car_info = response.css('.result-meta .result-price , .hdrlnk').css('::text').extract()
        # for i in range(len(car_price)):
        #     append_csv_file(car_year_name[i],car_price[i])
        for i in range(len(car_price)):
            items['car_year_name'] = car_year_name[i]
            items['car_price'] = car_price[i]
            if ',' in car_urls[i]:
                lst = ["https://images.craigslist.org/{}_300x300.jpg".format(i[2:]) for i in car_urls[i].split(',')]
                items['file_urls'] = lst
            else:
                items['file_urls'] = "https://images.craigslist.org/" + car_urls[i][2:] + "_300x300.jpg"
            yield items
        #print('{} - {}'. format(car_year_name,car_price))
        
        
        # for url in self.start_urls:
        #     next_page = url + '?s=' + str(ExploitSpider.page_number)
        #     if ExploitSpider.page_number < 3000:
        #         ExploitSpider.page_number = ExploitSpider.page_number + 120
        #         yield response.follow(next_page, callback = self.parse,dont_filter=True)

       
# def append_csv_file(car_year_name,car_price):
#     line = f"{car_year_name} - {car_price}\n"
#     if not os.path.exists(csv_file):
#         with open(csv_file, 'w') as _f:
#             _f.write(line)
#         return
#     with open(csv_file, 'a') as _f:
#         _f.write(line)
